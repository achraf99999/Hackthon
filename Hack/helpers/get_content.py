from pdf2image import convert_from_bytes
import base64
from io import BytesIO
import docx


def get_content_from_bytes(filename: str, content: bytes):
    if filename.endswith(('.doc', '.docx')):
        doc = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith('.pdf'):
        images = convert_from_bytes(content)
        buffered = BytesIO()
        images[0].save(buffered, format="JPEG", quality=30, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    elif filename.endswith('.txt'):
        return content.decode("utf-8")
    else:
        raise ValueError("File type not supported")