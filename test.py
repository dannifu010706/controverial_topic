import pandas as pd
import numpy as np
from pathlib import Path

def count_words(text):
    """Count words in text, handling None/empty values."""
    if not text or pd.isna(text):
        return 0
    return len(str(text).split())

def test_csv_content(csv_file="brit_wiki_fulltext20250702 (1).csv"):
    """Test and analyze the content of the generated article columns."""
    
    print("=" * 60)
    print("CSV COLUMN CONTENT TESTER")
    print("=" * 60)
    
    # Try different encodings
    encodings_to_try = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    df = None
    used_encoding = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            used_encoding = encoding
            print(f"✅ Successfully read CSV with encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            if encoding == encodings_to_try[-1]:
                print(f"❌ Failed to read CSV file: {e}")
                return
    
    if df is None:
        print(f"❌ Could not read CSV file with any encoding")
        return
    
    print(f"📊 CSV Info:")
    print(f"   - Total rows: {len(df)}")
    print(f"   - Total columns: {len(df.columns)}")
    print(f"   - Columns: {list(df.columns)}")
    print()
    
    # Find article columns (columns that end with '_article')
    article_columns = [col for col in df.columns if col.endswith('_article')]
    
    if not article_columns:
        print("⚠️  No article columns found (columns ending with '_article')")
        print("   Available columns:", list(df.columns))
        return
    
    print(f"📝 Found {len(article_columns)} article column(s):")
    for col in article_columns:
        print(f"   - {col}")
    print()
    
    # Analyze each article column
    for col in article_columns:
        print("-" * 50)
        print(f"ANALYZING COLUMN: {col}")
        print("-" * 50)
        
        # Basic statistics
        total_rows = len(df)
        non_empty_rows = df[col].notna().sum()
        empty_rows = total_rows - non_empty_rows
        
        print(f"📈 Basic Statistics:")
        print(f"   - Total rows: {total_rows}")
        print(f"   - Non-empty rows: {non_empty_rows}")
        print(f"   - Empty rows: {empty_rows}")
        print(f"   - Fill rate: {non_empty_rows/total_rows*100:.1f}%")
        print()
        
        if non_empty_rows == 0:
            print("⚠️  No content found in this column")
            continue
        
        # Content analysis
        non_empty_content = df[col].dropna()
        
        # Word count analysis
        word_counts = []
        char_counts = []
        error_count = 0
        
        for content in non_empty_content:
            content_str = str(content)
            if content_str.startswith("ERROR:"):
                error_count += 1
            else:
                word_counts.append(count_words(content_str))
                char_counts.append(len(content_str))
        
        if word_counts:
            print(f"📊 Content Statistics:")
            print(f"   - Articles with content: {len(word_counts)}")
            print(f"   - Articles with errors: {error_count}")
            print(f"   - Average word count: {np.mean(word_counts):.0f}")
            print(f"   - Min word count: {min(word_counts)}")
            print(f"   - Max word count: {max(word_counts)}")
            print(f"   - Average character count: {np.mean(char_counts):.0f}")
            print()
        
        # Show sample content
        print(f"📋 Sample Content (first 3 non-empty entries):")
        sample_count = 0
        for idx, content in df[col].items():
            if pd.notna(content) and str(content).strip() and sample_count < 3:
                content_str = str(content)
                topic_name = "Unknown"
                
                # Try to get topic name from other columns
                if 'title_in_wiki' in df.columns and pd.notna(df.at[idx, 'title_in_wiki']):
                    topic_name = df.at[idx, 'title_in_wiki']
                elif 'title_in_britannica' in df.columns and pd.notna(df.at[idx, 'title_in_britannica']):
                    topic_name = df.at[idx, 'title_in_britannica']
                
                print(f"\n   Row {idx + 1} - Topic: {topic_name}")
                print(f"   Word count: {count_words(content_str)}")
                print(f"   Character count: {len(content_str)}")
                
                if content_str.startswith("ERROR:"):
                    print(f"   ❌ Error: {content_str}")
                else:
                    # Show first 300 characters
                    preview = content_str[:300].replace('\n', '\\n').replace('\r', '\\r')
                    print(f"   Preview: {preview}...")
                
                sample_count += 1
        
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Overall summary
    total_articles = 0
    total_errors = 0
    total_words = 0
    
    for col in article_columns:
        non_empty_content = df[col].dropna()
        for content in non_empty_content:
            content_str = str(content)
            if content_str.startswith("ERROR:"):
                total_errors += 1
            else:
                total_articles += 1
                total_words += count_words(content_str)
    
    print(f"📊 Overall Statistics:")
    print(f"   - Total generated articles: {total_articles}")
    print(f"   - Total errors: {total_errors}")
    if total_articles > 0:
        print(f"   - Average words per article: {total_words/total_articles:.0f}")
        print(f"   - Total words generated: {total_words:,}")
    print(f"   - Success rate: {total_articles/(total_articles + total_errors)*100:.1f}%" if (total_articles + total_errors) > 0 else "   - Success rate: N/A")

def test_specific_row(csv_file="brit_wiki_fulltext20250702 (1).csv", row_index=0):
    """Test a specific row's content in detail."""
    
    print("=" * 60)
    print(f"DETAILED ROW ANALYSIS - Row {row_index}")
    print("=" * 60)
    
    # Read CSV
    encodings_to_try = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    df = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            break
        except:
            continue
    
    if df is None:
        print("❌ Could not read CSV file")
        return
    
    if row_index >= len(df):
        print(f"❌ Row index {row_index} is out of range. CSV has {len(df)} rows.")
        return
    
    row = df.iloc[row_index]
    
    # Show all column values for this row
    print(f"📋 All columns for row {row_index}:")
    for col in df.columns:
        value = row[col]
        if pd.isna(value):
            print(f"   {col}: <empty>")
        else:
            value_str = str(value)
            if len(value_str) > 100:
                print(f"   {col}: {value_str[:100]}... [{len(value_str)} chars]")
            else:
                print(f"   {col}: {value_str}")
    
    # Focus on article columns
    article_columns = [col for col in df.columns if col.endswith('_article')]
    
    for col in article_columns:
        print(f"\n" + "=" * 40)
        print(f"ARTICLE COLUMN: {col}")
        print("=" * 40)
        
        content = row[col]
        if pd.isna(content):
            print("❌ Content is empty/NaN")
        else:
            content_str = str(content)
            print(f"✅ Content found:")
            print(f"   - Character count: {len(content_str)}")
            print(f"   - Word count: {count_words(content_str)}")
            print(f"   - Starts with: {content_str[:50]}...")
            print(f"   - Ends with: ...{content_str[-50:]}")
            
            if content_str.startswith("ERROR:"):
                print(f"❌ This is an error message: {content_str}")
            else:
                print("✅ This appears to be valid article content")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CSV column content")
    parser.add_argument("--csv-file", type=str, default="brit_wiki_fulltext20250702 (1).csv", 
                       help="Path to CSV file")
    parser.add_argument("--row", type=int, help="Specific row to analyze in detail")
    
    args = parser.parse_args()
    
    if args.row is not None:
        test_specific_row(args.csv_file, args.row)
    else:
        test_csv_content(args.csv_file)