#!/bin/bash

# Configuration
PDF_FILE="samsul.pdf"  # Replace with your PDF file path
BASE_URL="http://localhost:5000"  # Replace with your API base URL
FILENAME="samsul"  # Replace with desired filename for the endpoint

# Upload the PDF using cURL (sending raw PDF data)
curl -X POST \
  --data-binary "@${PDF_FILE}" \
  -H "Content-Type: application/pdf" \
  "${BASE_URL}/upload_roi/${FILENAME}" \
  -v