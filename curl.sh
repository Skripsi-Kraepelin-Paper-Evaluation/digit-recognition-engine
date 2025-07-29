#!/bin/bash

# Configuration
PDF_FILE="example.pdf"  # Replace with your PDF file path
BASE_URL="http://localhost:5000"  # Replace with your API base URL
FILENAME="example"  # Replace with desired filename for the endpoint

# Additional form data
OCCUPACY_AND_ROLE="Software Engineer"
LAST_EDU="S1"  # Valid options: SD, SMP, SMA, D1, D2, D3, D4, S1, S2, S3, Lainnya
POB="Jakarta"  # Place of Birth
DOB="1990-01-15"  # Date of Birth (YYYY-MM-DD format)

# Upload the PDF using cURL with multipart form data
curl -X POST \
  -F "file=@${PDF_FILE}" \
  -F "occupacyAndRole=${OCCUPACY_AND_ROLE}" \
  -F "lastEdu=${LAST_EDU}" \
  -F "pob=${POB}" \
  -F "dob=${DOB}" \
  "${BASE_URL}/upload_roi/${FILENAME}" \
  -v

echo "Upload completed!"
echo "Metadata will be saved to: roi_result/metadata/${FILENAME}.json"