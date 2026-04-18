import pandas as pd

# Read the CSV
df = pd.read_csv('logs/council_phase4_results.csv')

print("=" * 80)
print("CLASSIFICATION REPORT - VERIFICATION")
print("=" * 80)

print(f"\n✓ Total rows: {len(df)}")
print(f"✓ predicted_emotion column: EXISTS" if 'predicted_emotion' in df.columns else "❌ predicted_emotion: MISSING")
print(f"✓ Actual_Emotion column: EXISTS" if 'Actual_Emotion' in df.columns else "❌ Actual_Emotion: MISSING")

print(f"\n📊 Data Summary:")
print(f"  Predicted emotions: {sorted(df['predicted_emotion'].unique())}")
print(f"  Actual emotions: {sorted(df['Actual_Emotion'].unique())}")
print(f"  Missing values in predicted_emotion: {df['predicted_emotion'].isna().sum()}")
print(f"  Missing values in Actual_Emotion: {df['Actual_Emotion'].isna().sum()}")

print(f"\n✅ CSV IS READY FOR CLASSIFICATION REPORT!")
print(f"\nYou can now use this code:")
print(f"  from sklearn.metrics import classification_report, confusion_matrix")
print(f"  df = pd.read_csv('logs/council_phase4_results.csv')")
print(f"  print(classification_report(df['Actual_Emotion'], df['predicted_emotion']))")
print(f"  print(confusion_matrix(df['Actual_Emotion'], df['predicted_emotion']))")
