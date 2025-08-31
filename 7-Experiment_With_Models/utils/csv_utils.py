import logging

EXPECTED_NUM_COLS = 16  # keep in sync with results.csv header

def assert_and_write(writer, row):
    """Write a row to CSV, padding/truncating to EXPECTED_NUM_COLS.
    Logs mismatch instead of raising so pipeline never crashes.
    """
    if len(row) != EXPECTED_NUM_COLS:
        logging.warning(
            f"[CSV] Column mismatch – expected {EXPECTED_NUM_COLS}, got {len(row)}. Row will be padded/truncated"
        )
        if len(row) < EXPECTED_NUM_COLS:
            row = row + [''] * (EXPECTED_NUM_COLS - len(row))
        else:
            row = row[:EXPECTED_NUM_COLS]
    writer.writerow(row) 