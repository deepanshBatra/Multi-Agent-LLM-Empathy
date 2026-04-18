import csv
import json

with open('logs/council_phase4_results.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    
print('=== DATA QUALITY ISSUES ===\n')

# Check encoding issues
encoding_issues = []
for i, row in enumerate(rows):
    if 'Â' in row['Utterance'] or '\ufffd' in row['Utterance']:
        encoding_issues.append(i)

if encoding_issues:
    print(f'❌ Encoding Issues (rows with corrupted characters): {encoding_issues}')
else:
    print("✓ No encoding issues detected")

# Check empty fields
print('\n=== Empty/Invalid Fields ===')
empty_found = False
for i, row in enumerate(rows):
    for col in row:
        if row[col].strip() == '' or row[col].strip() == '{}':
            print(f'⚠️  Row {i}, Column "{col}": EMPTY/EMPTY_JSON')
            empty_found = True

if not empty_found:
    print("✓ No empty fields")

# Check predicted vs actual mismatch
print('\n=== Predicted vs Actual Emotion (Accuracy) ===')
matches = 0
for i, row in enumerate(rows):
    pred = row['predicted_emotion'].lower().strip()
    actual = row['Actual_Emotion'].lower().strip()
    match = pred == actual
    symbol = '✓' if match else '❌'
    if match:
        matches += 1
    print(f'Row {i}: Predicted={pred:12} Actual={actual:12} {symbol}')

accuracy = (matches / len(rows)) * 100
print(f'\nAccuracy: {matches}/{len(rows)} = {accuracy:.1f}%')

# Check Predicted_Emotion_Raw vs predicted_emotion
print('\n=== Predicted_Emotion_Raw JSON vs predicted_emotion column ===')
for i, row in enumerate(rows):
    try:
        raw_json = json.loads(row['Predicted_Emotion_Raw'])
        json_pred = raw_json.get('predicted_emotion', 'N/A').lower()
        col_pred = row['predicted_emotion'].lower()
        match = json_pred == col_pred
        symbol = '✓' if match else '❌'
        print(f'Row {i}: JSON={json_pred:12} Column={col_pred:12} {symbol}')
    except:
        print(f'Row {i}: ERROR parsing JSON')

# Summary of column structure
print('\n=== COLUMN STRUCTURE ===')
print(f'Total rows: {len(rows)}')
print(f'Total columns: {len(rows[0])}')
print(f'Columns: {list(rows[0].keys())}')
