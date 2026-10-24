import io
import math
import threading

import pypdfium2 as pdfium


_render_lock = threading.Lock()
MAX_PREVIEW_EDGE = 1600


class PreviewBusyError(RuntimeError):
    pass


def render_pdf_page(contents: bytes, page_number: int) -> bytes:
    # PDFium is not thread-safe, including calls on separate documents.
    if not _render_lock.acquire(timeout=5):
        raise PreviewBusyError("PDF renderer is busy.")
    try:
        with pdfium.PdfDocument(contents) as document:
            if not 1 <= page_number <= len(document):
                raise ValueError("Page not found.")
            page = document[page_number - 1]
            try:
                width, height = page.get_size()
                if not all(
                    math.isfinite(value) and value > 0 for value in (width, height)
                ):
                    raise ValueError("Invalid PDF page dimensions.")
                scale = min(2.0, MAX_PREVIEW_EDGE / max(width, height))
                bitmap = page.render(scale=scale)
                try:
                    with bitmap.to_pil() as image:
                        output = io.BytesIO()
                        image.save(output, format="PNG")
                        return output.getvalue()
                finally:
                    bitmap.close()
            finally:
                page.close()
    except pdfium.PdfiumError as error:
        raise ValueError("Unable to render this PDF.") from error
    finally:
        _render_lock.release()
