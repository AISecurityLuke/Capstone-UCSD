import re
import html

def sanitize_text(text):
    text = html.unescape(text)  # Decode HTML entities
    text = re.sub(r'[\r\n\t]+', ' ', text)  # Normalize whitespace
    text = re.sub(r'(<script.*?>.*?</script>)', '[SCRIPT]', text, flags=re.DOTALL | re.IGNORECASE)  # Block scripts
    text = re.sub(r'\b(eval|exec|os\.system|subprocess|shutil|import)\b', r'[CODE:\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'[\u202E\u200F\u200E\u202A-\u202D]', '[RTL]', text)  # Neutralize RTL attacks
    return text.strip()

def escape_for_logging(text):
    if not isinstance(text, str):
        text = str(text)

    # Excel injection fix
    if text and text[0] in ('=', '+', '-', '@'):
        text = "'" + text

    # Shell/script symbol neutralization
    text = re.sub(r'[|;&`$><]', '[SYM]', text)

    # CRLF/log-breaking neutralization
    text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    # Dangerous code references
    text = re.sub(r'(__import__|eval|exec|getattr|setattr|subprocess|fetch|XMLHttpRequest)', '[DANGEROUS]', text, flags=re.IGNORECASE)

    return text