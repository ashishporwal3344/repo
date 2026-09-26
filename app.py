import io
import os
import re
import shutil
import tempfile
import zipfile
from urllib.parse import urlparse
from datetime import datetime

import fitz  # PyMuPDF
import gradio as gr
import requests
from PIL import Image


# =====================================================================
# 1. VALIDATION HELPERS
# =====================================================================

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def validate_url(url: str):
    """Return (is_valid, message) for a http/https URL."""
    if not url or not url.strip():
        return False, "URL cannot be empty."
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f"Could not parse URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return False, "URL must start with http:// or https://"
    if not parsed.netloc:
        return False, "URL is missing a domain (e.g. example.com)."
    return True, "URL is valid."


def validate_pdf(path: str):
    """Return (is_valid, message). Detects corrupt / password-protected PDFs."""
    if not os.path.isfile(path):
        return False, "File does not exist."
    if not path.lower().endswith(".pdf"):
        return False, "File is not a .pdf file."
    try:
        doc = fitz.open(path)
    except Exception as exc:
        return False, f"Corrupt or unreadable PDF: {exc}"
    try:
        if doc.is_encrypted:
            if not doc.authenticate(""):
                return False, "Password protected PDF."
        if doc.page_count == 0:
            return False, "PDF has no pages."
    finally:
        doc.close()
    return True, "PDF is valid."


def validate_image(path: str):
    if not path or not os.path.isfile(path):
        return False, "Image file does not exist."
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, f"Unsupported image format: {ext}. Use PNG, JPG/JPEG or WEBP."
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        return False, f"Invalid or corrupt image: {exc}"
    return True, "Image is valid."


# =====================================================================
# 1b. PDF-LINKS-FROM-TXT HELPERS (download PDFs from a list of URLs)
# =====================================================================

def parse_links_txt(path: str):
    """Read a .txt file and return a list of non-empty, non-comment lines
    (one PDF URL per line). Lines starting with '#' are treated as
    comments and skipped."""
    links = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            links.append(line)
    return links


def download_pdf_from_url(url: str, dest_folder: str, timeout: int = 30):
    """
    Download a PDF from `url` into `dest_folder`. Returns the local file
    path on success. Raises ValueError/requests exceptions on failure -
    callers should catch and turn these into a FAILED ProcessResult.
    """
    is_valid, msg = validate_url(url)
    if not is_valid:
        raise ValueError(msg)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; BulkPDFReplacer/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    # Some servers mislabel PDFs, so also allow through if the bytes
    # start with the PDF magic number even when Content-Type is wrong.
    looks_like_pdf = "pdf" in content_type or resp.content[:5] == b"%PDF-"
    if not looks_like_pdf:
        raise ValueError(f"URL did not return a PDF (Content-Type: {content_type or 'unknown'})")

    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or "downloaded.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, filename)
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
            counter += 1

    with open(dest_path, "wb") as f:
        f.write(resp.content)

    return dest_path


# =====================================================================
# 2. IMAGE HELPERS (dimensions, aspect-fit, format normalization)
# =====================================================================

def prepare_image_bytes(path: str):
    """Load any supported image and return (png_bytes, (width, height))."""
    with Image.open(path) as img:
        if img.mode in ("P", "LA"):
            img = img.convert("RGBA")
        elif img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue(), img.size


def calculate_fit_rect(target_rect, img_width, img_height, preserve_aspect=True):
    """Compute a centered, aspect-preserving (or exact-fill) rect inside target_rect."""
    tx0, ty0, tx1, ty1 = target_rect
    target_w = max(tx1 - tx0, 1e-6)
    target_h = max(ty1 - ty0, 1e-6)

    if not preserve_aspect or img_width <= 0 or img_height <= 0:
        return (tx0, ty0, tx1, ty1)

    img_ratio = img_width / img_height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_w = target_w
        new_h = target_w / img_ratio
    else:
        new_h = target_h
        new_w = target_h * img_ratio

    offset_x = (target_w - new_w) / 2
    offset_y = (target_h - new_h) / 2
    x0 = tx0 + offset_x
    y0 = ty0 + offset_y
    return (x0, y0, x0 + new_w, y0 + new_h)


# =====================================================================
# 3. PDF PROCESSING (link + first-page image replacement)
# =====================================================================

class ProcessResult:
    STATUS_OK = "OK"
    STATUS_WARNING = "WARNING"
    STATUS_FAILED = "FAILED"

    def __init__(self, filename, status, messages=None):
        self.filename = filename
        self.status = status
        self.messages = messages or []
        self.info = []  # informational notes that don't affect status (e.g. compression result)

    def add_warning(self, msg):
        self.messages.append(msg)
        if self.status == self.STATUS_OK:
            self.status = self.STATUS_WARNING

    def add_info(self, msg):
        self.info.append(msg)

    def __str__(self):
        info_suffix = f" ({'; '.join(self.info)})" if self.info else ""
        if self.status == self.STATUS_OK:
            return f"[OK] {self.filename}{info_suffix}"
        joined = "; ".join(self.messages)
        if self.status == self.STATUS_WARNING:
            return f"[WARNING] {self.filename} - {joined}{info_suffix}"
        return f"[FAILED] {self.filename}\nReason: {joined}"


def _get_first_page_images(page):
    """Return [{'xref','bbox','width','height','area'}, ...] for page 1's rendered images."""
    images = []
    try:
        info_list = page.get_image_info(xrefs=True)
    except Exception:
        info_list = []
    for info in info_list:
        xref = info.get("xref", 0)
        bbox = info.get("bbox")
        if not bbox or xref == 0:
            continue
        x0, y0, x1, y1 = bbox
        images.append({
            "xref": xref,
            "bbox": (x0, y0, x1, y1),
            "width": x1 - x0,
            "height": y1 - y0,
            "area": max(0.0, x1 - x0) * max(0.0, y1 - y0),
        })
    return images


def select_image(images, strategy="largest"):
    if not images:
        return None
    if strategy == "first":
        return images[0]
    return max(images, key=lambda im: im["area"])


def replace_first_page_image(page, image_descriptor, new_image_path, preserve_aspect=True):
    """
    Replace the detected image cleanly (old image is never left visible).

    - preserve_aspect=False: direct xref swap, fills the exact original box.
    - preserve_aspect=True (or direct swap unavailable): cover the old
      image with an opaque white rectangle, then insert the new image
      centered and fit (never stretched) inside the original box.
    """
    png_bytes, (img_w, img_h) = prepare_image_bytes(new_image_path)
    xref = image_descriptor["xref"]
    bbox = image_descriptor["bbox"]

    if not preserve_aspect:
        try:
            page.replace_image(xref, stream=png_bytes)
            return "direct"
        except Exception:
            pass

    rect = fitz.Rect(*bbox)
    page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
    target_rect = calculate_fit_rect(bbox, img_w, img_h, preserve_aspect=preserve_aspect)
    page.insert_image(fitz.Rect(*target_rect), stream=png_bytes, overlay=True)
    return "fallback"


def replace_page_links(page, new_url, replace_all=True):
    replaced = 0
    for link in page.get_links():
        if link.get("kind") == fitz.LINK_URI:
            link["uri"] = new_url
            page.update_link(link)
            replaced += 1
            if not replace_all:
                break
    return replaced


def insert_whole_page_image(page, new_image_path, preserve_aspect=True):
    """
    Insert `new_image_path` covering the page's entire first page.
    Used when a PDF has no existing first-page image to replace - this
    draws the image directly on top of whatever else is on the page.
    """
    png_bytes, (img_w, img_h) = prepare_image_bytes(new_image_path)
    page_rect = page.rect
    full_rect = (page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
    target_rect = calculate_fit_rect(full_rect, img_w, img_h, preserve_aspect=preserve_aspect)
    page.insert_image(fitz.Rect(*target_rect), stream=png_bytes, overlay=True)


def add_whole_page_link(page, url):
    """Add a clickable link annotation covering the page's entire area,
    pointing to `url`. This is additive - it does not remove or affect
    any existing link annotations on the page."""
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": page.rect,
        "uri": url,
    })


# =====================================================================
# 3b. PDF COMPRESSION (target a size range by recompressing images)
# =====================================================================

def _get_size_kb(path):
    return os.path.getsize(path) / 1024.0


def compress_pdf_to_target(pdf_path, min_kb=200, max_kb=400,
                            quality_steps=(85, 70, 55, 40, 30, 20, 15, 10)):
    """
    Recompress the PDF at `pdf_path` (in place) toward a [min_kb, max_kb]
    size window, by re-encoding its embedded images as JPEG at
    progressively lower quality until the file fits under max_kb (or the
    lowest quality step is reached).

    This only ever shrinks a file - if the PDF is already below min_kb
    (e.g. a short, mostly-text document), it is left alone; a PDF cannot
    be meaningfully "padded" up to a minimum without adding junk data,
    so min_kb here is used only to report status, not to inflate output.

    Returns (final_size_kb, status_message).
    """
    size_kb = _get_size_kb(pdf_path)

    # Baseline: lossless optimization only (garbage collection, stream
    # compression) - often enough on its own for text-heavy PDFs.
    tmp_path = pdf_path + ".tmp"
    doc = fitz.open(pdf_path)
    try:
        doc.save(tmp_path, garbage=4, deflate=True, clean=True)
    except Exception:
        doc.save(tmp_path)
    doc.close()
    shutil.move(tmp_path, pdf_path)
    size_kb = _get_size_kb(pdf_path)

    if size_kb <= max_kb:
        if size_kb < min_kb:
            return size_kb, f"below {min_kb:.0f}KB target after optimization (left as-is)"
        return size_kb, "within target range after lossless optimization"

    # Still too big - recompress embedded images at decreasing JPEG
    # quality until it fits, or we run out of quality steps.
    for quality in quality_steps:
        doc = fitz.open(pdf_path)
        try:
            for page in doc:
                for img in page.get_images(full=True):
                    xref = img[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha >= 4:  # CMYK/other -> RGB first
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        if pix.alpha:  # JPEG has no alpha channel
                            pix = fitz.Pixmap(pix, 0)
                        jpeg_bytes = pix.tobytes("jpeg", jpg_quality=quality)
                        page.replace_image(xref, stream=jpeg_bytes)
                    except Exception:
                        continue  # leave that particular image untouched

            tmp_path = pdf_path + ".tmp"
            try:
                doc.save(tmp_path, garbage=4, deflate=True, clean=True)
            except Exception:
                doc.save(tmp_path)
        finally:
            doc.close()

        shutil.move(tmp_path, pdf_path)
        size_kb = _get_size_kb(pdf_path)
        if size_kb <= max_kb:
            break

    if size_kb > max_kb:
        return size_kb, f"could not reach {max_kb:.0f}KB even at lowest quality (final {size_kb:.0f}KB)"
    if size_kb < min_kb:
        return size_kb, f"compressed to {size_kb:.0f}KB (below {min_kb:.0f}KB minimum)"
    return size_kb, f"compressed to {size_kb:.0f}KB (within target range)"


def process_single_pdf(pdf_path, output_path, new_url, image_path, options):
    filename = os.path.basename(pdf_path)
    result = ProcessResult(filename, ProcessResult.STATUS_OK)

    is_valid, msg = validate_pdf(pdf_path)
    if not is_valid:
        result.status = ProcessResult.STATUS_FAILED
        result.messages.append(msg)
        return result

    doc = None
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted and not doc.authenticate(""):
            result.status = ProcessResult.STATUS_FAILED
            result.messages.append("Password protected PDF.")
            return result
        if doc.page_count == 0:
            result.status = ProcessResult.STATUS_FAILED
            result.messages.append("PDF has no pages.")
            return result

        first_page = doc[0]

        if options.get("replace_links") and new_url:
            total = sum(
                replace_page_links(p, new_url, replace_all=options.get("replace_all_links", True))
                for p in doc
            )
            if options.get("whole_page_link", True):
                try:
                    add_whole_page_link(first_page, new_url)
                    if total == 0:
                        result.add_info("No existing hyperlink found - added a whole-page link on page 1")
                    else:
                        result.add_info(f"Replaced {total} existing link(s); also added a whole-page link on page 1")
                except Exception as exc:
                    result.add_warning(f"Could not add whole-page link: {exc}")
            elif total == 0:
                result.add_warning("No hyperlink found")

        if options.get("replace_image") and image_path:
            images = _get_first_page_images(first_page)
            if not images:
                if options.get("fill_missing_image", True):
                    try:
                        insert_whole_page_image(
                            first_page, image_path,
                            preserve_aspect=options.get("preserve_aspect", True),
                        )
                        result.add_info("No existing image found - inserted image covering the full first page")
                    except Exception as exc:
                        result.add_warning(f"Could not insert full-page image: {exc}")
                else:
                    result.add_warning("No image found on first page")
            else:
                chosen = select_image(images, strategy=options.get("image_selection", "largest"))
                try:
                    replace_first_page_image(
                        first_page, chosen, image_path,
                        preserve_aspect=options.get("preserve_aspect", True),
                    )
                except Exception as exc:
                    result.add_warning(f"Image replacement failed: {exc}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            doc.save(output_path, garbage=3, deflate=True, clean=True)
        except Exception:
            doc.save(output_path)
        doc.close()
        doc = None  # already closed above; avoid double-close in finally

        if options.get("compress"):
            min_kb = options.get("compress_min_kb", 200)
            max_kb = options.get("compress_max_kb", 400)
            try:
                final_kb, note = compress_pdf_to_target(output_path, min_kb=min_kb, max_kb=max_kb)
                if "could not reach" in note:
                    result.add_warning(f"Compression: {note}")
                else:
                    result.add_info(f"Compression: {note}")
            except Exception as exc:
                result.add_warning(f"Compression failed: {exc}")

        return result
    except Exception as exc:
        result.status = ProcessResult.STATUS_FAILED
        result.messages.append(str(exc))
        return result
    finally:
        if doc is not None:
            doc.close()


_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]+')


def build_output_path(pdf_path, output_folder, keep_original_filename=True,
                       custom_name=None, index=None, total_count=1):
    """
    Decide the output filename for one processed PDF.

    - custom_name (if given, non-empty) always wins: the file is named
      "<custom_name>.pdf" when there's only one PDF in the batch, or
      "<custom_name>_1.pdf", "<custom_name>_2.pdf", ... when there are
      several, using `index` (1-based) to number them.
    - Otherwise falls back to the original filename (optionally with a
      "_replaced" suffix), same as before.

    Either way, if the resulting path already exists, a numeric suffix
    is added so nothing already-written in this run gets overwritten.
    """
    if custom_name and custom_name.strip():
        base = _INVALID_FILENAME_CHARS.sub("_", custom_name.strip())
        base = base.rsplit(".pdf", 1)[0] if base.lower().endswith(".pdf") else base
        if not base:
            base = "output"
        if total_count > 1 and index is not None:
            filename = f"{base}_{index}.pdf"
        else:
            filename = f"{base}.pdf"
    else:
        filename = os.path.basename(pdf_path)
        if not keep_original_filename:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_replaced{ext}"

    candidate = os.path.join(output_folder, filename)
    if os.path.exists(candidate):
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(output_folder, f"{name}_{counter}{ext}")
            counter += 1
    return candidate


# =====================================================================
# 4. WEB UI (Gradio)
# =====================================================================

# IMPORTANT CHANGE FROM THE ORIGINAL SCRIPT:
# The old WORKDIR pointed at a hardcoded local Mac folder
# ("/Users/pankajyadav/Downloads/..."), which only exists on that one
# machine. On any server (Render, Railway, HF Spaces, etc.) that path
# doesn't exist and the app would crash. We now use the system's temp
# directory instead, which always exists and is always writable,
# wherever the app runs.
WORKDIR = os.path.join(tempfile.gettempdir(), "bulk_pdf_replacer_web")
OUTPUT_DIR = os.path.join(WORKDIR, "output")


def process_all(pdf_files, links_txt_file, image_file, new_url, replace_links, replace_image,
                 preserve_aspect, replace_all_links, keep_filenames, image_selection,
                 compress, compress_min_kb, compress_max_kb, custom_filename,
                 fill_missing_image, whole_page_link, custom_zip_name,
                 progress=gr.Progress()):
    """
    Main handler wired to the "START BULK REPLACEMENT" button.
    Yields (log_text, zip_file_path_or_None) so the log updates live in
    the UI as each PDF finishes.

    NOTE: because this is a generator (it uses `yield`), the Blocks app
    below MUST call demo.queue() - otherwise Gradio raises an error as
    soon as this handler is invoked.
    """
    log_lines = []

    def emit():
        return "\n".join(log_lines), None

    # gr.Number fields can come through as None (e.g. if the user clears
    # the box) - guard against that here so `compress_min_kb > compress_max_kb`
    # below can't raise "'>' not supported between NoneType and ...".
    compress_min_kb = 200 if compress_min_kb is None else compress_min_kb
    compress_max_kb = 400 if compress_max_kb is None else compress_max_kb

    pdf_files = list(pdf_files) if pdf_files else []
    links_txt_path = None
    if links_txt_file:
        links_txt_path = links_txt_file.name if hasattr(links_txt_file, "name") else links_txt_file

    if not pdf_files and not links_txt_path:
        log_lines.append("[FAILED] No PDF files uploaded and no links .txt file provided.")
        yield emit()
        return

    if replace_links:
        valid, msg = validate_url(new_url or "")
        if not valid:
            log_lines.append(f"[FAILED] Invalid URL: {msg}")
            yield emit()
            return

    if replace_image:
        if not image_file:
            log_lines.append("[FAILED] No replacement image uploaded.")
            yield emit()
            return
        valid, msg = validate_image(image_file)
        if not valid:
            log_lines.append(f"[FAILED] Invalid image: {msg}")
            yield emit()
            return

    if compress and compress_min_kb > compress_max_kb:
        log_lines.append("[FAILED] Compression min size cannot be greater than max size.")
        yield emit()
        return

    # Fresh workspace for this run
    if os.path.isdir(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    downloads_dir = os.path.join(WORKDIR, "downloaded")
    os.makedirs(downloads_dir, exist_ok=True)

    # A queue of (source_label, local_path_or_None) - local_path is None
    # for links that still need downloading, filled in as we go so
    # download failures show up in the live log immediately.
    pdf_sources = [(os.path.basename(pdf_file.name if hasattr(pdf_file, "name") else pdf_file),
                     pdf_file.name if hasattr(pdf_file, "name") else pdf_file)
                    for pdf_file in pdf_files]

    link_list = []
    if links_txt_path:
        try:
            link_list = parse_links_txt(links_txt_path)
            if not link_list:
                log_lines.append("[WARNING] Links .txt file was provided but contained no URLs.")
                yield emit()
        except Exception as exc:
            log_lines.append(f"[FAILED] Could not read links .txt file: {exc}")
            yield emit()
            return

    options = {
        "replace_image": replace_image,
        "replace_links": replace_links,
        "preserve_aspect": preserve_aspect,
        "replace_all_links": replace_all_links,
        "image_selection": image_selection,
        "compress": compress,
        "compress_min_kb": compress_min_kb,
        "compress_max_kb": compress_max_kb,
        "fill_missing_image": fill_missing_image,
        "whole_page_link": whole_page_link,
    }

    total = len(pdf_sources) + len(link_list)
    if total == 0:
        log_lines.append("[FAILED] Nothing to process.")
        yield emit()
        return

    ok = warnings = failed = 0
    results = []
    step = 0

    # --- Process directly-uploaded PDFs first ---
    for filename, pdf_path in pdf_sources:
        step += 1
        progress((step - 1) / total, desc=f"Processing {filename}")

        output_path = build_output_path(
            pdf_path, OUTPUT_DIR, keep_filenames,
            custom_name=custom_filename, index=step, total_count=total,
        )
        try:
            result = process_single_pdf(
                pdf_path, output_path,
                new_url if replace_links else None,
                image_file if replace_image else None,
                options,
            )
        except Exception as exc:
            result = ProcessResult(filename, ProcessResult.STATUS_FAILED, [str(exc)])

        results.append(result)
        if result.status == ProcessResult.STATUS_OK:
            ok += 1
        elif result.status == ProcessResult.STATUS_WARNING:
            warnings += 1
        else:
            failed += 1
        log_lines.append(str(result))
        yield emit()

    # --- Download and process PDFs from the links .txt file ---
    for url in link_list:
        step += 1
        progress((step - 1) / total, desc=f"Downloading {url}")
        display_name = url

        try:
            downloaded_path = download_pdf_from_url(url, downloads_dir)
        except Exception as exc:
            result = ProcessResult(display_name, ProcessResult.STATUS_FAILED, [f"Download failed: {exc}"])
            results.append(result)
            failed += 1
            log_lines.append(str(result))
            yield emit()
            continue

        filename = os.path.basename(downloaded_path)
        progress((step - 1) / total, desc=f"Processing {filename}")
        output_path = build_output_path(
            downloaded_path, OUTPUT_DIR, keep_filenames,
            custom_name=custom_filename, index=step, total_count=total,
        )
        try:
            result = process_single_pdf(
                downloaded_path, output_path,
                new_url if replace_links else None,
                image_file if replace_image else None,
                options,
            )
            # Make clear in the log which URL this file came from.
            result.filename = f"{filename} (from {url})"
        except Exception as exc:
            result = ProcessResult(f"{filename} (from {url})", ProcessResult.STATUS_FAILED, [str(exc)])

        results.append(result)
        if result.status == ProcessResult.STATUS_OK:
            ok += 1
        elif result.status == ProcessResult.STATUS_WARNING:
            warnings += 1
        else:
            failed += 1
        log_lines.append(str(result))
        yield emit()

    progress(1.0, desc="Finishing up")

    log_lines.append("")
    log_lines.append("Processing completed.")
    log_lines.append("")
    log_lines.append(f"Total PDFs: {total}")
    log_lines.append(f"Successful: {ok}")
    log_lines.append(f"Warnings: {warnings}")
    log_lines.append(f"Failed: {failed}")

    # Write the plain-text log file into the output folder too
    log_path = os.path.join(OUTPUT_DIR, "pdf_replacement_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Bulk PDF Link & Image Replacer - Log\n")
        f.write(f"Run at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        for r in results:
            f.write(str(r) + "\n")
        f.write(f"\nTotal PDFs: {total}\nSuccessful: {ok}\nWarnings: {warnings}\nFailed: {failed}\n")

    # Zip the output folder for a single download
    if custom_zip_name and custom_zip_name.strip():
        zip_base = _INVALID_FILENAME_CHARS.sub("_", custom_zip_name.strip())
        zip_base = zip_base.rsplit(".zip", 1)[0] if zip_base.lower().endswith(".zip") else zip_base
        zip_filename = f"{zip_base or 'output'}.zip"
    else:
        zip_filename = "bulk_pdf_replacer_output.zip"
    zip_path = os.path.join(WORKDIR, zip_filename)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, filenames in os.walk(OUTPUT_DIR):
            for fn in filenames:
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, OUTPUT_DIR))

    yield "\n".join(log_lines), zip_path


CUSTOM_CSS = """
#title {text-align: center;}
#start-btn {font-size: 16px; font-weight: bold; height: 48px;}
"""

# Hides the "Use via API" and "Built with Gradio" footer links, leaving
# only the Settings (gear) button visible at the bottom of the page.
HIDE_FOOTER_LINKS_JS = """
() => {
    const hideByText = () => {
        document.querySelectorAll('footer a, footer button').forEach(el => {
            const text = el.textContent.trim();
            if (text === 'Use via API' || text === 'Built with Gradio') {
                el.style.display = 'none';
            }
        });
    };
    hideByText();
    const interval = setInterval(hideByText, 500);
    setTimeout(() => clearInterval(interval), 5000);
}
"""

with gr.Blocks(title="Bulk PDF Link & Image Replacer") as demo:
    demo.load(None, None, None, js=HIDE_FOOTER_LINKS_JS)
    gr.Markdown("# 📎 Bulk PDF Link & Image Replacer", elem_id="title")
    gr.Markdown(
        "Upload PDFs directly, or provide a .txt file of PDF links to download automatically. "
        "Every PDF's hyperlinks and first-page image are replaced, optionally compressed to a "
        "target size, and saved separately — **originals are never modified**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. PDF Files")
            pdf_files = gr.File(
                label="Upload PDF files (multiple allowed)",
                file_count="multiple",
                file_types=[".pdf"],
            )
            links_txt_file = gr.File(
                label="Or: upload a .txt file with PDF links (one URL per line)",
                file_count="single",
                file_types=[".txt"],
            )

            gr.Markdown("### 2. New Link")
            new_url = gr.Textbox(
                label="Enter New URL",
                placeholder="https://example.com/new-page",
            )
            replace_all_links = gr.Checkbox(value=True, label="Replace all existing links")

            gr.Markdown("### 3. Replacement Image")
            image_file = gr.Image(
                label="Select Replacement Image (PNG / JPG / JPEG / WEBP)",
                type="filepath",
            )
            image_selection = gr.Radio(
                choices=[("Largest image", "largest"), ("First image", "first")],
                value="largest",
                label="If a PDF has multiple images on page 1, replace:",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 4. Processing Options")
            replace_image_cb = gr.Checkbox(value=True, label="Replace first-page image")
            replace_links_cb = gr.Checkbox(value=True, label="Replace PDF hyperlinks")
            preserve_aspect_cb = gr.Checkbox(value=True, label="Preserve image aspect ratio")
            keep_filenames_cb = gr.Checkbox(value=True, label="Keep original filenames")
            fill_missing_image_cb = gr.Checkbox(
                value=True,
                label="If a PDF has no first-page image, insert my image covering the whole first page",
            )
            whole_page_link_cb = gr.Checkbox(
                value=True,
                label="Add a link over the whole first page (every PDF, in addition to replacing existing links)",
            )

            gr.Markdown("### 4b. Compression")
            compress_cb = gr.Checkbox(value=True, label="Compress output PDFs to a target size")
            with gr.Row():
                compress_min_kb = gr.Number(value=200, label="Min size (KB)", precision=0)
                compress_max_kb = gr.Number(value=400, label="Max size (KB)", precision=0)

            gr.Markdown("### 4c. Custom Output Names")
            custom_filename_tb = gr.Textbox(
                label="Custom output filename (optional)",
                placeholder="e.g. my-report — leave blank to keep original filenames",
            )
            gr.Markdown(
                "If set, replaces original filenames: one PDF -> `<name>.pdf`; "
                "multiple PDFs -> `<name>_1.pdf`, `<name>_2.pdf`, etc. "
                "Leave blank to keep each file's original name (uses 'Keep original "
                "filenames' below instead)."
            )
            custom_zip_name_tb = gr.Textbox(
                label="Custom zip filename (optional)",
                placeholder="e.g. my-batch — leave blank for the default name",
            )

            gr.Markdown("### 5. Start Processing")
            start_btn = gr.Button("START BULK REPLACEMENT", elem_id="start-btn", variant="primary")

            gr.Markdown("### 6. Processing Log")
            log_box = gr.Textbox(label="Log", lines=16, interactive=True)

            gr.Markdown("### 7. Download Results")
            output_zip = gr.File(label="Download Output (.zip)")

    start_btn.click(
        fn=process_all,
        inputs=[
            pdf_files, links_txt_file, image_file, new_url,
            replace_links_cb, replace_image_cb, preserve_aspect_cb,
            replace_all_links, keep_filenames_cb, image_selection,
            compress_cb, compress_min_kb, compress_max_kb, custom_filename_tb,
            fill_missing_image_cb, whole_page_link_cb, custom_zip_name_tb,
        ],
        outputs=[log_box, output_zip],
    )

demo.queue()


if __name__ == "__main__":
    # IMPORTANT NOTES:
    #
    # 1. share=True IS needed here even though Render already gives a
    #    public URL. On startup, Gradio pings its own 127.0.0.1 address
    #    to confirm the server is reachable; inside Render's container
    #    that self-ping fails, and Gradio refuses to start unless
    #    share=True is set (it then also opens a gradio.live tunnel,
    #    which is harmless extra - just use your onrender.com URL, not
    #    the gradio.live one, since the onrender.com one is permanent).
    #
    # 2. server_name="0.0.0.0" - binds to all network interfaces so the
    #    hosting platform's reverse proxy can reach the app. Binding to
    #    127.0.0.1 (the default) only accepts connections from the same
    #    machine, which breaks on a server.
    #
    # 3. server_port=int(os.environ.get("PORT", 7860)) - most platforms
    #    (Render included) tell your app which port to listen on via the
    #    PORT environment variable at runtime. Falls back to 7860 for
    #    local runs.
    #
    # 4. allowed_paths=[WORKDIR] is kept - required so Gradio will serve
    #    the generated zip file back to the browser for download.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        allowed_paths=[WORKDIR],
        share=True,
    )
