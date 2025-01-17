import pdfplumber
import pandas as pd
import json
import re
from fuzzywuzzy import fuzz
import fitz
from decimal import Decimal, getcontext
from collections import Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast
from fuzzywuzzy import fuzz
import re


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


def extract_cols(words_df, coordinate_param, round_param, x_threshold=3, y_threshold=15):
    '''
    Extract columns based on x0 or x1 coordinates and calculate min and max positions.
    Iteratively include elements with similar x1 values and expand column boundaries from the bottom.

    Input:
    words_df (DataFrame): containing words and their positions.
    coordinate_param (str): The coordinate to group by ('x0' or 'x1').
    round_param (int): The degree to which the coordinates should be rounded.
    x_threshold (float): The threshold for considering x1 values as similar.
    y_threshold (float): The threshold for considering elements close to current boundaries.

    Output:
    result_df (DataFrame): containing the group names (x0 or x1), min_bottom, and max_top for each group.
    '''

    words_df2 = words_df.copy()

    # filter alignment in lines starts
    words_df2 = words_df2[words_df2['x0'] > 50]

    # Round to detect alignment
    words_df2[coordinate_param] = round(
        words_df2[coordinate_param], round_param)

    grouped = words_df2.groupby(coordinate_param)

    # Initialize lists to store the results
    min_bottoms = []
    max_tops = []
    group_names = []

    # Iterate over each group
    for name, group in grouped:
        # Filter group with at least n_lines (kant 4)
        n_lines = 2
        if len(group) >= n_lines:
            # Initialize min_bottom and max_top
            min_bottom = group['bottom'].max()
            max_top = group['top'].min()
            mean_x1 = group['x1'].mean()

            # Iteratively expand boundaries from the bottom
            while True:
                # Find elements with similar x1 and close to current bottom boundary
                similar_x1 = words_df2[
                    (abs(words_df2['x1'] - mean_x1) <= x_threshold) &
                    (abs(words_df2['bottom'] - min_bottom) <= y_threshold) &
                    # Only consider elements below or at the current bottom
                    (words_df2['bottom'] >= min_bottom)
                ]

                # Check if we found new elements
                new_min_bottom = similar_x1['bottom'].max(
                ) if not similar_x1.empty else min_bottom

                # If no change in bottom boundary, break the loop
                if new_min_bottom == min_bottom:
                    break

                # Update bottom boundary and mean_x1
                min_bottom = new_min_bottom
                mean_x1 = similar_x1['x1'].mean(
                ) if not similar_x1.empty else mean_x1

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


def extract_table(selected_p, content, mt):
    '''
    Extract table data from a page.

    Input:
    selected_p (int): The selected page number.
    content (DataFrame): The content of the page.
    mt (float): Margin top value.

    Output:
    list_of_table_df (list): List of DataFrames containing table data.
    list_of_bbox (list): List of bounding boxes for the tables.
    selected_p (int): The selected page number.
    '''
    df = content
    page_text_df = content

    bounding_box = extract_cols(page_text_df, 'x1', 2)

    # Compute margin_top
    df['margin_top'] = df['bottom'] - df['top'].shift(1)
    df['margin_top'] = df['margin_top'].fillna(0)
    df['margin_top'] = df.apply(lambda row: 0 if row['top'] == df['top'].shift(
        1).loc[row.name] else row['margin_top'], axis=1)

    gap_y_df = df[df['margin_top'] > 27]
    combined_data = np.concatenate(
        (bounding_box[['min_bottom']], bounding_box[['max_top']]), axis=0)

    if 'combined_data' in locals() and combined_data.shape[0] > 0:
        scaler = MinMaxScaler()
        normalized_combined_data = scaler.fit_transform(combined_data)
        split_index = len(bounding_box)
        normalized_min_bottom = normalized_combined_data[:split_index]
        normalized_max_top = normalized_combined_data[split_index:]
        bounding_box['min_bottom_normalized'] = np.round(
            normalized_min_bottom, 0)
        bounding_box['max_top_normalized'] = np.round(normalized_max_top, 0)

        grouped = bounding_box.groupby(['min_bottom_normalized'])
        list_of_dfs = [group.reset_index(drop=True) for _, group in grouped]

        list_of_top_bottom_bbox = []
        for _, group in grouped:
            if len(group) > 1:
                list_of_top_bottom_bbox.append({
                    'bbox_top': float(group['max_top'].min()),
                    'bbox_bottom': float(group['min_bottom'].max())
                })

        list_of_bbox = []
        list_of_table_df = []
        for top_bottom_bbox in list_of_top_bottom_bbox:
            bbox_top, bbox_bottom = top_bottom_bbox['bbox_top'], top_bottom_bbox['bbox_bottom']
            gap_y_df['top_diff'] = np.abs(gap_y_df['top'] - bbox_top)
            gap_y_df['bottom_diff'] = np.abs(gap_y_df['bottom'] - bbox_bottom)
            gap_y_df['total_diff'] = gap_y_df['top_diff'] + \
                gap_y_df['bottom_diff']
            nearest_row = gap_y_df.loc[gap_y_df['total_diff'].idxmin()]
            nearest_row['top']
            gap_y_df['top_diff'] = np.abs(gap_y_df['top'] - bbox_top)
            nearest_top_row = gap_y_df.loc[gap_y_df['top_diff'].idxmin()]
            nearest_top_df = gap_y_df[gap_y_df['top']
                                      == nearest_top_row['top']]
            bbox_top_final, bbox_bottom_final = float(
                nearest_top_df['top']), float(bbox_bottom)
            padding = 5

            target_phrases = ['World total, natural and synthetic',
                              'World total (ilmenite and rutile, rounded)', 'World total (rounded)', 'World total']

            # Filter the dataframe to include only rows within the current bounding box
            temp_df = page_text_df[(page_text_df['top'] >= bbox_top_final - padding) &
                                   (page_text_df['bottom'] <= bbox_bottom_final + padding)]

            comb_temp_df = complexe_word(temp_df, 0)

            comb_temp_df.to_csv('sample_comb_df.csv', index=False)

            # Find rows containing target phrases
            for phrase in target_phrases:
                target_rows = comb_temp_df[comb_temp_df['text'].str.contains(
                    phrase, case=False, na=False, regex=False)]
                if not target_rows.empty:
                    # Update bbox_bottom_final to the bottom of the first detected phrase
                    bbox_bottom_final = float(target_rows['bottom'].min())
                    break  # Stop after finding the first match

            table_df = page_text_df[(page_text_df['top'] >= bbox_top_final - padding) & (
                page_text_df['bottom'] <= bbox_bottom_final + padding)]

            # New code to modify table_df based on the specified conditions
            threshold = 200  # You may need to adjust this value, before 120

            # Step 1: Find the leftmost elements for each row (unique 'bottom' value)

            def group_by_bottom_with_tolerance(bottom, tolerance=2):
                return np.round(bottom / tolerance) * tolerance

            leftmost_elements = table_df.groupby(lambda x: group_by_bottom_with_tolerance(
                table_df.loc[x, 'bottom'])).apply(lambda x: x.loc[x['x0'].idxmin()])

            # Step 2: Check which of these leftmost elements have x0 > threshold
            eligible_elements = leftmost_elements[leftmost_elements['x0'] > threshold]

            if not eligible_elements.empty:
                # Step 3: Among eligible elements, find the one with minimum top
                elem = eligible_elements.loc[eligible_elements['top'].idxmin()]

                table_df = table_df[table_df['top'] >= elem['top'] - mt]

                # Update bbox_top_final
                bbox_top_final = float(elem['top'])

            bbox_start, bbox_end = float(
                table_df['x0'].min()), float(table_df['x1'].max())

            list_of_bbox.append({
                'bbox_top': bbox_top_final - mt,
                'bbox_bottom': bbox_bottom_final,
                'bbox_start': bbox_start,
                'bbox_end': bbox_end
            })
            list_of_table_df.append(table_df)

        return list_of_table_df, list_of_bbox, selected_p
    else:
        print("Dataframe is either not defined or empty. Scaling not applied.")
        return [], [], selected_p


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
                # the top coordinate of the word relative to the entire document
                w1['doctop'] = min(w1['doctop'], w2['doctop'])
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


def get_duplicated_values(input_list):
    '''
    Find duplicated values in a list.

    Input:
    input_list (list): List of values to check for duplicates.

    Output:
    list: List of values that appear more than once in the input list.
    '''
    count = Counter(input_list)
    return [item for item, freq in count.items() if freq > 1]


def extract_largest_text(pdf_path, page):
    '''extract the titles from pages 34 to 205 

    Extract the largest text elements (titles) from a specified page in a PDF document.

    Input:
    pdf_path (str): The file path to the PDF document.
    page (int): The page number from which to extract the largest text elements (0-indexed).

    Output:
    max_size_elements_df (DataFrame): containing the text elements with the largest font size on the specified page.
    '''
    words_df = extract_word_positions(pdf_path, page)

    max_size = words_df['height'].max()
    max_size_elements_df = words_df[(words_df['height'] <= max_size) &
                                    (words_df['height'] > max_size - 1.6) &
                                    (words_df['bottom'] <= 70)]

    merged_max_size_elements_df = complexe_word(
        max_size_elements_df, round_param=1)

    merged_max_size_elements_df['page'] = page

    return merged_max_size_elements_df


def match_element_in_text(materials, title):
    """
    Match the material to the titles

    Input:
    materials (list): List of material names to search for.
    title (str): The title text to search in.

    Output:
    str or None: The matched material name if found, else None.
    """
    title = title.lower()  # Convert text to lowercase for case-insensitive matching
    for material in materials:
        if re.search(re.escape(material.lower()), title):
            return material
    return None


# to change!!! 7yed dik abrasssive, then element_name!
def extract_positions_for_elements(pdf_path, material_pages):
    """
      Extract the positions of text elements from specified pages in a PDF document.

      Input:
      pdf_path (str): The file path to the PDF document.
      material_pages (dict): A dictionary where the keys are material names and the values are dictionaries 
                             containing 'title' as a DataFrame and 'pages' as a list of page numbers.

      Output:
      content (dict): A dictionary where the keys are material names, and the values contain:
                      - 'material_title': The title DataFrame.
                      - 'pages_num': A list of page numbers.
                      - 'pages_content': A list of DataFrames containing the content for each page.
                      - 'remarks': A list of remarks (initially empty).
                      - 'tables': A list of tables (initially empty).
      """
    content = {}

    for material, data in material_pages.items():
        title = data['title']
        pages = data['pages']

        content[material] = {
            'material_title': title,  # Store the entire DataFrame here
            'pages_num': pages,
            'pages_content': [],
            'remarks': [],
            'tables': []
        }

        for page_number in pages:
            page_content = extract_word_positions(pdf_path, page_number - 1)
            content[material]['pages_content'].append(page_content)

    return content


def most_repeated_value(lst):
    """
    Find the most frequently occurring element in a list.

    Input:
    lst (list): The input list to analyze.

    Output:
    The most common element in the list.
    """
    # Use Counter to count the frequency of each element
    counter = Counter(lst)
    # Find the most common element
    most_common_element = counter.most_common(1)[0][0]
    return most_common_element


def belongs_to_same_group(row1, row2):
    """
    Check if two rows belong to the same group based on their x-coordinates and bottom position.

    Input:
    row1, row2 (Series): Two rows from a DataFrame containing 'x0', 'x1', and 'bottom' coordinates.

    Output:
    bool: True if the rows belong to the same group, False otherwise.
    """
    # Check if intervals [x0, x1] of row1 and row2 overlap
    interval1_start, interval1_end = min(
        row1['x0'], row1['x1']), max(row1['x0'], row1['x1'])
    interval2_start, interval2_end = min(
        row2['x0'], row2['x1']), max(row2['x0'], row2['x1'])

    intervals_overlap = not (
        interval1_end < interval2_start or interval2_end < interval1_start)
    condition = intervals_overlap and row1['bottom'] != row2['bottom']
    return condition


def custom_outer_merge(df1, df2, iteration):
    """
    Perform a custom outer merge on two DataFrames based on the 'bottom' coordinate.

    Input:
    df1, df2 (DataFrame): The two DataFrames to merge.
    iteration (int): The current iteration number, used for naming the new text column.

    Output:
    DataFrame: The merged DataFrame.
    """
    merged = []
    used_indices_df2 = set()
    for i, row1 in df1.iterrows():
        match = False
        for j, row2 in df2.iterrows():
            if abs(row1['bottom'] - row2['bottom']) <= 5 and j not in used_indices_df2:
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
            combined_row = {
                col: None for col in df1.columns if col.startswith('text')}
            combined_row['bottom'] = row2['bottom']
            combined_row[f'text_{iteration}'] = row2['text']
            merged.append(combined_row)

    return pd.DataFrame(merged)


def merge_list_of_dataframes(dfs):
    """
    Merge a list of DataFrames based on their 'text' and 'bottom' columns.

    Input:
    dfs (list): A list of DataFrames to merge.

    Output:
    DataFrame: The merged DataFrame.
    """
    if not dfs:
        return pd.DataFrame()

    merged_df = dfs[0][['text', 'bottom']]
    for i in range(1, len(dfs)):
        merged_df = custom_outer_merge(
            merged_df, dfs[i][['text', 'bottom']], i)

    return merged_df


def spot_indice(list_of_table_df, list_of_bbox, selected_p, pdf_path):
    """
    Identify and extract indices (smaller text elements) within table bounding boxes.

    Input:
    list_of_table_df (list): List of DataFrames containing table data.
    list_of_bbox (list): List of bounding box dictionaries for each table.
    selected_p (int): The selected page number.
    pdf_path (str): The file path to the PDF document.

    Output:
    list: A list of dictionaries containing table DataFrames, bounding boxes, and identified indices.
    """
    list_table_and_bbox = list(zip(list_of_table_df, list_of_bbox))

    indice_bbox_table = []
    for table_df, bbox in list_table_and_bbox:
        with pdfplumber.open(pdf_path) as pdf:

            page_ch = pdf.pages[selected_p]
            chars = page_ch.chars

            # Filter words that fall within the specified coordinates
            chars_in_area = [
                char for char in chars
                if bbox['bbox_top'] - 1.3 <= char['top'] and bbox['bbox_bottom'] >= char['bottom']
            ]

            list_of_indices = []
            for char in chars_in_area:
                # 5? 1 , depends on the situation
                if char['height'] < table_df['height'].mode()[0] - 0.01:
                    char_left = {key: char[key] for key in [
                        'text', 'x0', 'x1', 'bottom', 'top', 'height', 'width']}
                    list_of_indices.append(char_left)

            indice_bbox_table.append({
                'table_df': table_df,
                'bbox': bbox,
                'indices': list_of_indices
            })

    return indice_bbox_table


def update_words_coordinates(indice_bbox_table):
    """
    Update word coordinates in tables based on identified indices.

    Input:
    indice_bbox_table (list): A list of dictionaries containing table DataFrames, bounding boxes, and indices.

    Output:
    list: The updated indice_bbox_table with adjusted word coordinates.
    """
    for entry in indice_bbox_table:
        words_df = entry['table_df']
        bbox = entry['bbox']
        chars_in_area = entry['indices']

        # Iterate over each character in the chars_in_area
        for char in chars_in_area:
            char_x0, char_top, char_x1, char_bottom = char['x0'], char['top'], char['x1'], char['bottom']

            # Find the word that contains this character
            for idx, word_row in words_df.iterrows():
                word_x0, word_top, word_x1, word_bottom = word_row[
                    'x0'], word_row['top'], word_row['x1'], word_row['bottom']

                if (word_x0 <= char_x0 <= word_x1 and word_top <= char_top <= word_bottom):
                    word_text = word_row['text']
                    char_text = char['text']

                    # Check if the character is on the left or right boundary of the word
                    if char_x0 <= word_x0:  # Character is to the left
                        # Remove the first occurrence of the character
                        word_text = word_text.replace(char_text, '', 1)
                        # Update the word text and coordinates
                        words_df.at[idx, 'text'] = word_text
                        words_df.at[idx, 'x0'] = char_x1
                    elif char_x1 >= word_x1:  # Character is to the right
                        # Remove the last occurrence of the character
                        word_text = word_text[::-
                                              1].replace(char_text[::-1], '', 1)[::-1]
                        # Update the word text and coordinates
                        words_df.at[idx, 'text'] = word_text
                        words_df.at[idx, 'x1'] = char_x0

                    # Update the width of the word
                    words_df.at[idx, 'width'] = words_df.at[idx,
                                                            'x1'] - words_df.at[idx, 'x0']

                    break  # Break after finding and processing the matching word

    return indice_bbox_table


def table_to_df(list_of_table_df, list_of_bbox, selected_p, pdf_path):
    """
    Convert extracted table data to structured DataFrames.

    Input:
    list_of_table_df (list): List of DataFrames containing raw table data.
    list_of_bbox (list): List of bounding boxes for each table.
    selected_p (int): Selected page number.
    pdf_path (str): Path to the PDF file.

    Output:
    list: List of processed and merged DataFrames for each table.
    """
    indice_bbox_table = spot_indice(
        list_of_table_df, list_of_bbox, selected_p, pdf_path)
    indice_bbox_table = update_words_coordinates(indice_bbox_table)

    for i in range(len(indice_bbox_table)):
        list_of_table_df[i] = indice_bbox_table[i]['table_df']

    for i in range(len(list_of_table_df)):
        table_df = list_of_table_df[i]
        mode_height = table_df['height'].mode()[0] - 3
        list_of_table_df[i] = table_df[table_df['height']
                                       >= round(mode_height, 0)]

    merged_df_list = []
    for df in list_of_table_df:
        comb_table_df = complexe_word(df, 0)
        groups = []

        for i, row1 in comb_table_df.iterrows():
            added_to_any_group = False
            for group in groups:
                if any(belongs_to_same_group(row1, comb_table_df.iloc[j]) for j in group):
                    group.append(i)
                    added_to_any_group = True
            if not added_to_any_group:
                groups.append([i])

        comb_table_df['group'] = -1
        group_label = 0
        for group in groups:
            for index in group:
                if comb_table_df.at[index, 'group'] == -1:
                    comb_table_df.at[index, 'group'] = group_label
            group_label += 1
#
        dff = comb_table_df[comb_table_df['group'] == 0]

        # Sort the data by 'bottom'
        data_sorted = dff.sort_values(by='bottom')

        # Group the data by 'bottom' and create new columns for each text in the same line
        grouped = data_sorted.groupby('bottom')['text'].apply(
            lambda x: x.reset_index(drop=True)).unstack().reset_index()
        # Calculate the minimum x0 value for each group (line)
        min_x0 = data_sorted.groupby(
            'bottom')['x0'].min().reset_index(name='min_x0')

        # Merge the min_x0 values back into the grouped dataframe
        grouped_with_min_x0 = pd.merge(grouped, min_x0, on='bottom')
        # Fill NaN values with the first non-NaN value of each row
        filled_grouped = grouped_with_min_x0.apply(
            lambda row: row.ffill(axis=0).bfill(axis=0), axis=1)

        # Replace 'NaN' strings with the first value in each row
        # Exclude 'bottom' and 'min_x0' columns
        for col in filled_grouped.columns[1:-1]:
            filled_grouped[col] = filled_grouped[col].replace(
                'NaN', method='ffill').replace('NaN', method='bfill')

        grps = comb_table_df['group'].unique().tolist()

        df_min_x0_tuples = []
        for i, grp in enumerate(grps):
            df_grp = comb_table_df[comb_table_df['group'] == grp]
            if not df_grp['bottom'].is_unique:
                data_sorted = df_grp.sort_values(by='bottom')

                # Group the data by 'bottom' and create new columns for each text in the same line
                grouped = data_sorted.groupby('bottom')['text'].apply(
                    lambda x: x.reset_index(drop=True)).unstack().reset_index()
                # Calculate the minimum x0 value for each group (line)
                min_x0 = data_sorted.groupby(
                    'bottom')['x0'].min().reset_index(name='min_x0')

                # Merge the min_x0 values back into the grouped dataframe
                grouped_with_min_x0 = pd.merge(grouped, min_x0, on='bottom')
                # Fill NaN values with the first non-NaN value of each row
                filled_grouped = grouped_with_min_x0.apply(
                    lambda row: row.ffill(axis=0).bfill(axis=0), axis=1)

                # Replace 'NaN' strings with the first value in each row
                # Exclude 'bottom' and 'min_x0' columns
                for col in filled_grouped.columns[1:-1]:
                    filled_grouped[col] = filled_grouped[col].replace(
                        'NaN', method='ffill').replace('NaN', method='bfill')

                min_x0 = filled_grouped['min_x0'].min()
                result_df = filled_grouped.sort_values(by='bottom')
                result_df = result_df.drop(columns=['min_x0'])
                for col in result_df.columns.tolist():
                    if col != 'bottom':
                        result_df_col = result_df[['bottom', col]].rename(
                            columns={col: 'text'})
                        df_min_x0_tuples.append((result_df_col, min_x0))

            else:
                min_x0 = df_grp['x0'].min()
                result_df = df_grp.sort_values(by='bottom')[['text', 'bottom']]
                df_min_x0_tuples.append((result_df, min_x0))

        # Sort the list of tuples by min_x0
        df_min_x0_tuples_sorted = sorted(df_min_x0_tuples, key=lambda x: x[1])

        # Extract the sorted DataFrames into a list
        dfs = [df_tuple[0] for df_tuple in df_min_x0_tuples_sorted]

        merged_df = merge_list_of_dataframes(dfs)
    #
        merged_df_list.append(merged_df)

    return merged_df_list


def extract_text_between_delimiters(df, pdf_path, page, bottom=82):
    '''
    Extract text within parentheses and brackets from specified sections of a PDF document along with their bounding boxes.

    Input:
    df (DataFrame): DataFrame containing text elements and their positions.
    pdf_path (str): The file path to the PDF document.
    page (int): The page number from which to extract the text (0-indexed).
    bottom (int, optional): The bottom coordinate threshold for filtering text elements. Default is 82.

    Output:
    result_df (DataFrame): DataFrame containing the text found within parentheses and brackets and their bounding boxes.
    '''

    # Assuming extract_largest_text is a predefined function that extracts the largest text element on the page
    title_bottom = extract_largest_text(pdf_path, page)['bottom'].max()

    # Filter the elements based on the top value
    filtered_elements = df[
        (df['top'] < bottom) &
        (df['top'] > title_bottom)
    ]

    # Extract text within parentheses and brackets from the filtered elements
    rows = []
    capture_parentheses = False
    capture_brackets = False
    current_text = ""
    min_x0, max_x1, min_top, max_bottom = float('inf'), 0, float('inf'), 0
    directions, heights, widths = set(), set(), set()

    for index, row in filtered_elements.iterrows():
        text = row['text']
        if '(' in text:
            capture_parentheses = True
        if '[' in text:
            capture_brackets = True
        if capture_parentheses or capture_brackets:
            current_text += text + " "
            min_x0 = min(min_x0, row['x0'])
            max_x1 = max(max_x1, row['x1'])
            min_top = min(min_top, row['top'])
            max_bottom = max(max_bottom, row['bottom'])
            directions.add(row['direction'])
            heights.add(row['height'])
            widths.add(row['width'])
        if ')' in text and capture_parentheses:
            capture_parentheses = False
            rows.append({
                'text': current_text.strip(),
                'x0': min_x0,
                'x1': max_x1,
                'top': min_top,
                'bottom': max_bottom,
                'doctop': None,  # doctop can be set to None or computed if needed
                'upright': True,  # Assuming upright is always True
                'height': max(heights),  # Assuming you want the maximum height
                'width': sum(widths),  # Sum of widths, if they are contiguous
                'direction': directions.pop() if len(directions) == 1 else list(directions),
                'page': page
            })
            current_text = ""
            min_x0, max_x1, min_top, max_bottom = float(
                'inf'), 0, float('inf'), 0
            directions, heights, widths = set(), set(), set()
        if ']' in text and capture_brackets:
            capture_brackets = False
            rows.append({
                'text': current_text.strip(),
                'x0': min_x0,
                'x1': max_x1,
                'top': min_top,
                'bottom': max_bottom,
                'doctop': None,  # doctop can be set to None or computed if needed
                'upright': True,  # Assuming upright is always True
                'height': max(heights),  # Assuming you want the maximum height
                'width': sum(widths),  # Sum of widths, if they are contiguous
                'direction': directions.pop() if len(directions) == 1 else list(directions),
                'page': page
            })
            current_text = ""
            min_x0, max_x1, min_top, max_bottom = float(
                'inf'), 0, float('inf'), 0
            directions, heights, widths = set(), set(), set()

    # Check for elements that share the same bottom value and concatenate their texts
    filtered_elements = filtered_elements.sort_values(by=['bottom', 'top'])
    current_bottom = None
    current_row = None

    for index, row in filtered_elements.iterrows():
        if current_bottom is None or row['bottom'] != current_bottom:
            if current_row:
                rows.append(current_row)
            current_bottom = row['bottom']
            current_row = {
                'text': row['text'],
                'x0': row['x0'],
                'x1': row['x1'],
                'top': row['top'],
                'bottom': row['bottom'],
                'doctop': None,
                'upright': True,
                'height': row['height'],
                'width': row['width'],
                'direction': row['direction'],
                'page': page
            }
        else:
            current_row['text'] += " " + row['text']
            current_row['x1'] = max(current_row['x1'], row['x1'])
            current_row['bottom'] = max(current_row['bottom'], row['bottom'])
            current_row['width'] += row['width']

    if current_row:
        rows.append(current_row)

    # Consolidate global remarks only
    global_remarks = []
    for row in rows:
        if ('(' in row['text'] and ')' in row['text']) or ('[' in row['text'] and ']' in row['text']):
            global_remarks.append(row)

    result_df = pd.DataFrame(global_remarks)

    # Further combine text if it belongs to the same remark but got split due to line breaks
    combined_remarks = []
    i = 0
    while i < len(global_remarks):
        current = global_remarks[i]
        if '(' in current['text'] and ')' not in current['text']:
            # Combine until we find the closing parenthesis
            j = i + 1
            while j < len(global_remarks) and ')' not in global_remarks[j]['text']:
                current['text'] += " " + global_remarks[j]['text']
                current['x1'] = max(current['x1'], global_remarks[j]['x1'])
                current['bottom'] = max(
                    current['bottom'], global_remarks[j]['bottom'])
                j += 1
            if j < len(global_remarks) and ')' in global_remarks[j]['text']:
                current['text'] += " " + global_remarks[j]['text']
                current['x1'] = max(current['x1'], global_remarks[j]['x1'])
                current['bottom'] = max(
                    current['bottom'], global_remarks[j]['bottom'])
            i = j
        elif '[' in current['text'] and ']' not in current['text']:
            # Combine until we find the closing bracket
            j = i + 1
            while j < len(global_remarks) and ']' not in global_remarks[j]['text']:
                current['text'] += " " + global_remarks[j]['text']
                current['x1'] = max(current['x1'], global_remarks[j]['x1'])
                current['bottom'] = max(
                    current['bottom'], global_remarks[j]['bottom'])
                j += 1
            if j < len(global_remarks) and ']' in global_remarks[j]['text']:
                current['text'] += " " + global_remarks[j]['text']
                current['x1'] = max(current['x1'], global_remarks[j]['x1'])
                current['bottom'] = max(
                    current['bottom'], global_remarks[j]['bottom'])
            i = j
        combined_remarks.append(current)
        i += 1

    result_df = pd.DataFrame(combined_remarks)
    return result_df

    # Open the existing PDF
    # document = fitz.open(input_pdf_path)

    # # Iterate over each material
    # for material, data in scraping_base.items():
    #     material_title_df = pd.DataFrame(data['material_title'])
    #     pages = data['pages_num']
    #     pages_content_df = pd.concat([pd.DataFrame(page_content) for page_content in data['pages_content']])
    #     remarks_df = pd.DataFrame(data['remarks'])

    #     # Iterate over each specified page for the material
    #     for page_num in pages:
    #         page = document[page_num - 1]  # Pages are 0-indexed in PyMuPDF

    #         # Draw rectangles over titles
    #         for index, row in material_title_df.iterrows():
    #             # Define the coordinates for the rectangle
    #             rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])

    #             # Draw the rectangle on the page
    #             page.draw_rect(rect, width=1, color=(1, 0, 0))  # Red rectangle for titles, width=1

    #         # Draw rectangles over pages content
    #         page_content = pages_content_df[(pages_content_df['doctop'] >= page_num * 1000) & (pages_content_df['doctop'] < (page_num + 1) * 1000)]
    #         for index, row in page_content.iterrows():
    #             # Define the coordinates for the rectangle
    #             rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])

    #             # Draw the rectangle on the page
    #             page.draw_rect(rect, width=1, color=(0, 0, 1))  # Blue rectangle for pages content, width=1

    #         # Draw rectangles over remarks on the correct page
    #         page_remarks = remarks_df[remarks_df['page'] == page_num]
    #         for index, row in page_remarks.iterrows():
    #             # Define the coordinates for the rectangle
    #             rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])

    #             # Draw the rectangle on the page
    #             page.draw_rect(rect, width=1, color=(0, 1, 0))  # Green rectangle for remarks, width=1

    # # Save the modified PDF to a new file
    # document.save(output_pdf_path)
    # document.close()


def draw_rectangles_for_materials(input_pdf_path, output_pdf_path, scraping_base, padding=2):
    """
    Draw rectangles on a PDF to highlight materials, remarks, and tables.

    Input:
    input_pdf_path (str): Path to the input PDF file.
    output_pdf_path (str): Path where the output PDF file will be saved.
    scraping_base (dict): Dictionary containing information about materials, remarks, and tables.
    padding (int): Padding for the rectangles. Default is 2.

    Output:
    None. The function saves a new PDF with drawn rectangles.
    """
    # Open the existing PDF
    document = fitz.open(input_pdf_path)

    def draw_rectangle(page, x0, top, x1, bottom, color, fill_color=None):
        """Helper function to draw a rectangle on the given page with padding."""
        rect = fitz.Rect(x0 - padding, top - padding,
                         x1 + padding, bottom + padding)
        if fill_color:
            page.draw_rect(rect, color=color, fill=fill_color, overlay=False)
        else:
            page.draw_rect(rect, color=color, overlay=False)

    # visualization purpose
    def draw_vertical_line(page, x):
        """Helper function to draw a vertical line on the given page."""
        start_point = fitz.Point(x, 0)
        end_point = fitz.Point(x, page.rect.height)
        page.draw_line(start_point, end_point, color=(1, 0, 0))  # Red line
    # visualization purpose

    def process_material_titles(material_title_df, document):
        """Process and draw rectangles over material titles."""
        for _, row in material_title_df.iterrows():
            page_num = row['page']
            page = document[page_num]  # Pages are 0-indexed in PyMuPDF
            draw_rectangle(page, row['x0'], row['top'], row['x1'], row['bottom'], color=(
                1, 0, 0), fill_color=(1, 0, 0, 0.))

    def process_remarks(remarks_df, page_num, document):
        """Process and draw rectangles over remarks."""
        page_remarks = remarks_df[remarks_df['page'] == page_num]
        page = document[page_num - 1]
        for _, row in page_remarks.iterrows():
            draw_rectangle(page, row['x0'], row['top'], row['x1'], row['bottom'], color=(
                0, 1, 0), fill_color=(0, 1, 0, 0.1))

    def process_tables(tables, page_num, document):
        """Process and draw rectangles over tables."""
        page = document[page_num - 1]
        for table, bbox, table_page in tables:
            if table_page == page_num - 1:
                draw_rectangle(page, bbox['bbox_start'], bbox['bbox_top'], bbox['bbox_end'],
                               bbox['bbox_bottom'], color=(0, 0, 1), fill_color=(0, 0, 1, 0.1))

    # visualization purpose
    for page in document:
        draw_vertical_line(page, 200)
    # visualization purpose

    # Iterate over each material
    for material, data in scraping_base.items():
        material_title_df = pd.DataFrame(data['material_title'])
        pages = data['pages_num']
        pages_content_df = pd.concat(
            [pd.DataFrame(page_content) for page_content in data['pages_content']])
        remarks_df = pd.DataFrame(data['remarks'])

        # Iterate over each specified page for the material
        for page_num in pages:
            process_material_titles(material_title_df, document)
            process_remarks(remarks_df, page_num, document)
            process_tables(data['tables'], page_num, document)

    # Save the modified PDF
    document.save(output_pdf_path)
    document.close()


def split_dataframe(df, split_indices):
    """
    Split a DataFrame into multiple DataFrames based on column indices.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        split_indices (list): List of column indices where to split the DataFrame.

    Returns:
        list: A list of DataFrames, each containing a subset of columns from the original DataFrame.
    """
    dfs = []
    start_col = 0
    for index in split_indices:
        dfs.append(df.iloc[:, start_col:index])
        start_col = index + 1
    # Append the last segment
    dfs.append(df.iloc[:, start_col:])
    return dfs


def convert_to_serializable(obj):
    """
    Convert pandas DataFrame or Series to a serializable format.

    Args:
        obj: The object to convert (expected to be a DataFrame, Series, or other).

    Returns:
        A serializable version of the input object.
    """
    if isinstance(obj, pd.DataFrame):
        # Convert DataFrame to list of dictionaries
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_list()  # Convert Series to list
    return obj  # Return the object as is if it's not a DataFrame or Series

# Recursively traverse and convert DataFrames in the dictionary


def serialize_dict(d):
    """
    Recursively traverse a dictionary and convert any DataFrames to serializable format.

    Args:
        d (dict): The dictionary to serialize.

    Returns:
        None. The dictionary is modified in place.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            serialize_dict(value)  # Recurse into nested dictionaries
        else:
            d[key] = convert_to_serializable(value)


def find_all_chemical_compositions(text, materials_list, reference_db):
    """
    Find all chemical compositions based on the text.

    Args:
        text (str): Text containing material names.
        materials_list (list): List of material names to match.
        reference_db (pd.DataFrame): DataFrame containing chemical compositions.

    Returns:
        list: List of chemical compositions if any materials are found, else None.
    """
    matched_materials = []
    for material in materials_list:
        if re.search(re.escape(material.lower()), text.lower()):
            matched_materials.append(material)
    compositions = []
    for material in matched_materials:
        filtered_df = reference_db[reference_db['sub_material_name'] == material]
        if not filtered_df.empty:
            comp = filtered_df['chemical_composition'].values[0]
            compositions.append(comp)
    return compositions if compositions else None


def find_metric_conversion_factor(text, metrics_list, metric_conversion_df):
    """
    Find the metric conversion factor based on the text.

    Args:
        text (str): Text containing metric units.
        metrics_list (list): List of metric units to match.
        metric_conversion_df (pd.DataFrame): DataFrame containing metric conversion factors.

    Returns:
        float or str: The conversion factor if a metric unit is found, else None.
    """
    for metric in metrics_list:
        if re.search(re.escape(metric.lower()), text.lower()):
            factor = metric_conversion_df[metric_conversion_df['Metric Mentioned']
                                          == metric]['conversion_factor']
            if not factor.empty:
                return factor.values[0]
    return None


def convert_dict_to_df(d):
    """
    Recursively convert a nested dictionary or list of dictionaries to DataFrame(s).

    Args:
        d: A dictionary, list of dictionaries, or other object.

    Returns:
        pd.DataFrame, list, or dict: Converted object(s).
    """
    if isinstance(d, list) and all(isinstance(i, dict) for i in d):
        return pd.DataFrame(d)
    elif isinstance(d, list):
        return [convert_dict_to_df(i) for i in d]
    elif isinstance(d, dict):
        return {k: convert_dict_to_df(v) for k, v in d.items()}
    else:
        return d


def convert_string_to_dict(input_data):
    """
    Convert string representations of dictionaries to actual dictionaries.
    Also handles nested structures like lists and dictionaries.

    Args:
        input_data: The input data to convert.

    Returns:
        The converted data structure.
    """
    if input_data is None:
        return None

    if isinstance(input_data, str):
        try:
            # Remove the percentage sign and convert to float
            item_dict = ast.literal_eval(input_data)
            return {k: float(v)for k, v in item_dict.items()}
        except (ValueError, SyntaxError):
            # If conversion fails, return the original string
            return input_data

    if isinstance(input_data, list):
        return [convert_string_to_dict(item) for item in input_data]

    if isinstance(input_data, dict):
        return {k: convert_string_to_dict(v) for k, v in input_data.items()}

    # If it's neither None, str, list, nor dict, return as is
    return input_data


def clean_numeric(val):
    """
    Clean a value and convert it to a numeric type if possible.
    
    Args:
        val: The value to clean, can be a string or any other type.
    
    Returns:
        float: The cleaned numeric value.
        np.nan: If the cleaned value is empty or not numeric.
    """
    if isinstance(val, str):
        # Remove commas and any other non-numeric characters except for decimal points
        cleaned = re.sub(r'[^\d.]+', '', val)
        return float(cleaned) if cleaned else np.nan
    return val


def match_composition(input_name, reference_df):
    """
    Find the best matching material composition from a reference DataFrame.
    
    Args:
        input_name (str): The name of the material to match.
        reference_df (pd.DataFrame): DataFrame containing reference material names and compositions.
    
    Returns:
        str: The matched composition if a good match is found (ratio > 80).
        None: If no good match is found.
    """
    input_name = str(input_name).lower()
    best_match = None
    best_ratio = 0
    for material in reference_df['sub_material_name']:
        ratio = fuzz.partial_ratio(input_name, str(material).lower())
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = material
    if best_ratio < 80:
        return None
    composition = reference_df.loc[reference_df['sub_material_name'] == best_match, 'chemical_composition'].iloc[0]
    return composition


def clean_numeric(val):
    if isinstance(val, str):
        cleaned = re.sub(r'[^\d.]+', '', val)
        return float(cleaned) if cleaned else np.nan
    return val


def convert_year_to_float(year):
    """
    Convert a year value to a float.
    
    Args:
        year: The year value, can be a string, int, or float.
    
    Returns:
        float: The year as a float.
        np.nan: If the year couldn't be converted.
    """
    if isinstance(year, str):
        year = year.strip().lower()
        if year.endswith('e'):
            return float(year[:-1])
        try:
            return float(year)
        except ValueError:
            return np.nan
    return int(year) if pd.notnull(year) else np.nan


def process_column(df, text_df, title_row, index, item, years):
    """
    Process a column of data, applying chemical composition percentages to the values.
    
    Args:
        df (pd.DataFrame): The DataFrame to update with new columns.
        text_df (pd.DataFrame): The DataFrame containing the text data.
        title_row (list): List of column titles.
        index (int): The index of the current column in title_row.
        item (dict): Dictionary containing material and chemical composition information.
        years (list): List of years to process.
    """
    for year in years:
        year_float = convert_year_to_float(year)
        if pd.notna(year_float):
            chem_comp_dict = item['chem_comp']
            if chem_comp_dict:
                for elem, percentage in chem_comp_dict.items():
                    new_col_name = f"{elem}_{year_float}"
                    try:
                        numeric_col = text_df.iloc[2:][title_row[index]].apply(clean_numeric)
                        result = numeric_col * percentage
                        result = result.replace([np.inf, -np.inf], np.nan)
                        if new_col_name in df.columns:
                            df[new_col_name] += result
                        else:
                            df[new_col_name] = pd.Series(index=df.index)
                            df.loc[df.index[2:], new_col_name] = result.values
                        print(f"Created/Updated column: {new_col_name}")
                    except Exception as e:
                        print(f"🔴Error processing column {title_row[index]}: {str(e)}")
            else:
                print(f"🔴 No valid chemical composition for {item['material']}. Keeping original data.")


def process_varied_composition(df, text_df, title_row, sheet_material, first_row, second_row):
    """
    Process columns with varied composition, creating new columns for each year.
    
    Args:
        df (pd.DataFrame): The DataFrame to update with new columns.
        text_df (pd.DataFrame): The DataFrame containing the text data.
        title_row (list): List of column titles.
        sheet_material (str): The name of the material for the current sheet.
        first_row (pd.Series): The first row of the DataFrame, typically containing material names.
        second_row (pd.Series): The second row of the DataFrame, typically containing years.
    """
    for index, col in enumerate(title_row):
        if text_df[col].iloc[2:].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().any():
            years = second_row.iloc[index].split(',') if isinstance(second_row.iloc[index], str) else [second_row.iloc[index]]
            for year in years:
                year_float = convert_year_to_float(year)
                if pd.notna(year_float):
                    new_col_name = f"{sheet_material}_{year_float}"
                    try:
                        numeric_col = text_df.iloc[2:][col].apply(clean_numeric)
                        if new_col_name in df.columns:
                            df[new_col_name] += numeric_col
                        else:
                            df[new_col_name] = pd.Series(index=df.index)
                            df.loc[df.index[2:], new_col_name] = numeric_col.values
                        print(f"Created/Updated column: {new_col_name}")
                    except Exception as e:
                        print(f"🔴Error processing column {col}: {str(e)}")