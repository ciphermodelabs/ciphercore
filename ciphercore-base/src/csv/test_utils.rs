use crate::errors::Result;

fn assert_rows_eq(rows: Vec<csv::StringRecord>, expected_rows: Vec<Vec<&str>>) {
    assert_eq!(rows.len(), expected_rows.len());
    for (i, (csv_row, expected_row)) in rows.into_iter().zip(expected_rows.into_iter()).enumerate()
    {
        assert_eq!(csv_row, expected_row, "row {i}");
    }
}

pub fn assert_table_eq(
    csv_bytes: Vec<u8>,
    expected_headers: Vec<&str>,
    expected_records: Vec<Vec<&str>>,
) -> Result<()> {
    let mut csv = csv::Reader::from_reader(csv_bytes.as_slice());
    assert_eq!(csv.headers()?, expected_headers);
    let csv_rows = csv
        .records()
        .collect::<std::result::Result<Vec<csv::StringRecord>, csv::Error>>()?;
    assert_rows_eq(csv_rows, expected_records);
    Ok(())
}

pub fn assert_sorted_table_eq(
    csv_bytes: Vec<u8>,
    expected_headers: Vec<&str>,
    expected_records: Vec<Vec<&str>>,
) -> Result<()> {
    let mut csv = csv::Reader::from_reader(csv_bytes.as_slice());
    assert_eq!(csv.headers()?, expected_headers);
    let mut csv_rows = csv
        .records()
        .collect::<std::result::Result<Vec<csv::StringRecord>, csv::Error>>()?;
    csv_rows.sort_by(|r1, r2: &csv::StringRecord| r1[0].cmp(&r2[0]));
    assert_rows_eq(csv_rows, expected_records);
    Ok(())
}

pub fn assert_table_unordered_eq(
    csv_bytes: Vec<u8>,
    expected_headers: Vec<&str>,
    expected_records: Vec<Vec<&str>>,
) -> Result<()> {
    let mut csv = csv::Reader::from_reader(csv_bytes.as_slice());
    assert_eq!(csv.headers()?, expected_headers);
    let mut csv_rows = csv
        .records()
        .collect::<std::result::Result<Vec<csv::StringRecord>, csv::Error>>()?;
    csv_rows.sort_by_key(|row| {
        row.clone()
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>()
    });
    let mut expected_records = expected_records;
    expected_records.sort();
    assert_rows_eq(csv_rows, expected_records);
    Ok(())
}
