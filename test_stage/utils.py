import pandas as pd
import pdfplumber
from decimal import Decimal, getcontext
from collections import Counter


def extract_word_positions(pdf_path, i):
    '''
    Extract words with their positions from a specific page of a PDF file.
    
    Input:
    pdf_path (str): The file path to the PDF document.
    i (int): The page number to extract words from (0-indexed).
    
    Output:
    words_df (DataFrame): DataFrame containing words and their positions on the specified page.
    '''
    
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[i]
        words = page.extract_words()
        words_df = pd.DataFrame(words)
    return words_df


def extract_positions_for_elements(pdf_path, material_pages):
    '''
      Extract the positions of text elements from specified pages in a PDF document.

      Input:
      pdf_path (str): The file path to the PDF document.
      material_pages (dict): A dictionary where the keys are material names and the values are lists of page numbers.

      Output:
      content (dict): A dictionary where the keys are material names, and the values contain the material name, pages, and content for each page.
    '''

    content = {}

    for material, pages in material_pages.items():
        content[material] = {
            'element_name': material,
            'pages': pages,
        }
        for i, page_number in enumerate(pages):
            page_content = extract_word_positions(pdf_path, page_number - 1)
            content[material][f'content_{i+1}'] = page_content

    return content


def extract_cols(words_df, coordinate_param, round_param, threshold=0.1):
    '''
    Extract columns based on x0 or x1 coordinates and calculate min and max positions.

    Input:
    words_df (DataFrame): containing words and their positions.
    coordinate_param (str): The coordinate to group by ('x0' or 'x1').
    round_param (int): The degree to which the coordinates should be rounded.

    Output:
    result_df (DataFrame): containing the group names (x0 or x1), min_bottom, and max_top for each group.
    '''

    words_df2 = words_df.copy()
        
    # filter alignment in lines starts
    words_df2 = words_df2[words_df2['x0'] > 50]

    words_df2[coordinate_param] = (words_df2[coordinate_param] // threshold) * threshold
    
    # Round to detect alignment
    words_df2[coordinate_param] = round(words_df2[coordinate_param], round_param)
    
    grouped = words_df2.groupby(coordinate_param)

    # Initialize lists to store the results
    min_bottoms = []    
    max_tops = []
    group_names = []

    # Iterate over each group
    for name, group in grouped:
        # Filter group with at least n_lines
        n_lines = 4
        if len(group) >= n_lines:
            # Calculate the max of 'bottom' and min of 'top' for the group
            min_bottom = group['bottom'].max()
            max_top = group['top'].min()

            # Append the results to the lists
            group_names.append(name)
            min_bottoms.append(min_bottom)
            max_tops.append(max_top)


    result_df = pd.DataFrame({
        coordinate_param: group_names,
        'min_bottom': min_bottoms,
        'max_top': max_tops
    })

    return result_df


def get_duplicated_values(input_list):
    count = Counter(input_list)
    return [item for item, freq in count.items() if freq > 1]


def complexe_word(words_df, round_param):

    '''
    Merge adjacent words in a DataFrame if they meet certain criteria.

    Input:
    words_df (DataFrame): DataFrame containing words and their positions.
    round_param (int): The degree to which the coordinates should be rounded.

    Output:
    merged_df (DataFrame): A DataFrame containing merged words and their updated positions.
    '''

    words_df2 = words_df.copy()
    words_df2['x0'] = words_df2['x0'].round(round_param)
    words_df2['x1'] = words_df2['x1'].round(round_param)

    merged_words = []
    i = 0

    while i < len(words_df2):
        w1 = words_df2.iloc[i].to_dict()
        j = i + 1

        while j < len(words_df2):
            w2 = words_df2.iloc[j].to_dict()
            if w1['x0'] < w2['x0'] and w2['x0'] - w1['x1'] < 4 and w1['bottom'] == w2['bottom']:
                # Merge w1 and w2
                w1['text'] = f"{w1['text']} {w2['text']}"
                w1['x1'] = w2['x1']
                w1['top'] = min(w1['top'], w2['top'])
                w1['doctop'] = min(w1['doctop'], w2['doctop'])  # the top coordinate of the word relative to the entire document 
                w1['bottom'] = max(w1['bottom'], w2['bottom'])
                w1['upright'] = w1['upright'] and w2['upright']
                w1['height'] = max(w1['height'], w2['height'])
                w1['width'] = w2['x1'] - w1['x0']
                w1['direction'] = w1['direction']
                j += 1
            else:
                break

        merged_words.append(w1)
        i = j

    merged_df = pd.DataFrame(merged_words)
    return merged_df


def most_repeated_value(lst):
    # Use Counter to count the frequency of each element
    counter = Counter(lst)
    # Find the most common element
    most_common_element = counter.most_common(1)[0][0]
    return most_common_element

def belongs_to_same_group(row1, row2):
    # Check if intervals [x0, x1] of row1 and row2 overlap
    interval1_start, interval1_end = min(row1['x0'], row1['x1']), max(row1['x0'], row1['x1'])
    interval2_start, interval2_end = min(row2['x0'], row2['x1']), max(row2['x0'], row2['x1'])
    
    intervals_overlap = not (interval1_end < interval2_start or interval2_end < interval1_start)
    condition = intervals_overlap and row1['bottom'] != row2['bottom']
    return condition

# Concatenate all DataFrames in the to_merge list
def custom_outer_merge(df1, df2, iteration):
    merged = []
    used_indices_df2 = set()
    for i, row1 in df1.iterrows():
        match = False
        for j, row2 in df2.iterrows():
            if abs(row1['bottom'] - row2['bottom']) <= 10 and j not in used_indices_df2:
                combined_row = row1.to_dict()
                combined_row[f'text_{iteration}'] = row2['text']
                merged.append(combined_row)
                used_indices_df2.add(j)
                match = True
                break
        if not match:
            combined_row = row1.to_dict()
            combined_row[f'text_{iteration}'] = None
            merged.append(combined_row)
    
    for j, row2 in df2.iterrows():
        if j not in used_indices_df2:
            combined_row = {col: None for col in df1.columns if col.startswith('text')}
            combined_row['bottom'] = row2['bottom']
            combined_row[f'text_{iteration}'] = row2['text']
            merged.append(combined_row)
    
    return pd.DataFrame(merged)

def merge_list_of_dataframes(dfs):
    if not dfs:
        return pd.DataFrame()

    merged_df = dfs[0][['text', 'bottom']]
    for i in range(1, len(dfs)):
        merged_df = custom_outer_merge(merged_df, dfs[i][['text', 'bottom']], i)

    return merged_df