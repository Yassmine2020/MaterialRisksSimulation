import pdfplumber
import pandas as pd
import json
import re
from fuzzywuzzy import fuzz
import fitz

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

def extract_cols(words_df, coordinate_param, round_param):
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
        
    words_df2 = words_df2[words_df2['x0'] > 50]
    words_df2['x0'] = round(words_df2['x0'], round_param)
    
    if coordinate_param == 'x0':
        # based on x0:
        grouped = words_df2.groupby('x0')

    elif coordinate_param == 'x1':
        grouped = words_df2.groupby('x1')

        # Initialize lists to store the results
    min_bottoms = []
    max_tops = []
    group_names = []

    # Iterate over each group
    for name, group in grouped:
        if len(group) > 3:
            # Calculate the min of 'bottom' and max of 'top' for the group
            min_bottom = group['bottom'].max()
            max_top = group['top'].min()

            # Append the results to the lists
            group_names.append(name)
            min_bottoms.append(min_bottom)
            max_tops.append(max_top)

    # Create a new DataFrame with the results
    result_df = pd.DataFrame({
        coordinate_param: group_names,
        'min_bottom': min_bottoms,
        'max_top': max_tops
    })

    return result_df

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
    max_size_elements_df = words_df[(words_df['height'] <= max_size) &\
                                    (words_df['height'] > max_size - 1.6) &\
                                    (words_df['bottom'] <= 70)]

    merged_max_size_elements_df = complexe_word(max_size_elements_df, round_param=1)

    merged_max_size_elements_df['page'] = page 

    return merged_max_size_elements_df

def match_element_in_text(materials, title):
    """
    Match the material to the titles
    """
    title = title.lower()  # Convert text to lowercase for case-insensitive matching
    for material in materials:
        if re.search(re.escape(material.lower()), title):
            return material
    return None

def extract_positions_for_elements(pdf_path, material_pages):   ## to change!!! 7yed dik abrasssive, then element_name!
    
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
            min_x0, max_x1, min_top, max_bottom = float('inf'), 0, float('inf'), 0
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
            min_x0, max_x1, min_top, max_bottom = float('inf'), 0, float('inf'), 0
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
                current['bottom'] = max(current['bottom'], global_remarks[j]['bottom'])
                j += 1
            if j < len(global_remarks) and ')' in global_remarks[j]['text']:
                current['text'] += " " + global_remarks[j]['text']
                current['x1'] = max(current['x1'], global_remarks[j]['x1'])
                current['bottom'] = max(current['bottom'], global_remarks[j]['bottom'])
            i = j
        elif '[' in current['text'] and ']' not in current['text']:
            # Combine until we find the closing bracket
            j = i + 1
            while j < len(global_remarks) and ']' not in global_remarks[j]['text']:
                current['text'] += " " + global_remarks[j]['text']
                current['x1'] = max(current['x1'], global_remarks[j]['x1'])
                current['bottom'] = max(current['bottom'], global_remarks[j]['bottom'])
                j += 1
            if j < len(global_remarks) and ']' in global_remarks[j]['text']:
                current['text'] += " " + global_remarks[j]['text']
                current['x1'] = max(current['x1'], global_remarks[j]['x1'])
                current['bottom'] = max(current['bottom'], global_remarks[j]['bottom'])
            i = j
        combined_remarks.append(current)
        i += 1

    result_df = pd.DataFrame(combined_remarks)
    return result_df

    # Open the existing PDF
    document = fitz.open(input_pdf_path)
    
    # Iterate over each material
    for material, data in scraping_base.items():
        material_title_df = pd.DataFrame(data['material_title'])
        pages = data['pages_num']
        pages_content_df = pd.concat([pd.DataFrame(page_content) for page_content in data['pages_content']])
        remarks_df = pd.DataFrame(data['remarks'])
        
        # Iterate over each specified page for the material
        for page_num in pages:
            page = document[page_num - 1]  # Pages are 0-indexed in PyMuPDF
            
            # Draw rectangles over titles
            for index, row in material_title_df.iterrows():
                # Define the coordinates for the rectangle
                rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])
                
                # Draw the rectangle on the page
                page.draw_rect(rect, width=1, color=(1, 0, 0))  # Red rectangle for titles, width=1

            # Draw rectangles over pages content
            page_content = pages_content_df[(pages_content_df['doctop'] >= page_num * 1000) & (pages_content_df['doctop'] < (page_num + 1) * 1000)]
            for index, row in page_content.iterrows():
                # Define the coordinates for the rectangle
                rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])
                
                # Draw the rectangle on the page
                page.draw_rect(rect, width=1, color=(0, 0, 1))  # Blue rectangle for pages content, width=1
            
            # Draw rectangles over remarks on the correct page
            page_remarks = remarks_df[remarks_df['page'] == page_num]
            for index, row in page_remarks.iterrows():
                # Define the coordinates for the rectangle
                rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])
                
                # Draw the rectangle on the page
                page.draw_rect(rect, width=1, color=(0, 1, 0))  # Green rectangle for remarks, width=1

    # Save the modified PDF to a new file
    document.save(output_pdf_path)
    document.close()

def draw_rectangles_for_materials(input_pdf_path, output_pdf_path, scraping_base):
    # Open the existing PDF
    document = fitz.open(input_pdf_path)
    
    # Iterate over each material
    for material, data in scraping_base.items():
        material_title_df = pd.DataFrame(data['material_title'])
        pages = data['pages_num']
        pages_content_df = pd.concat([pd.DataFrame(page_content) for page_content in data['pages_content']])
        remarks_df = pd.DataFrame(data['remarks'])
        
        # Iterate over each specified page for the material
        for page_num in pages:
            page = document[page_num - 1]  # Pages are 0-indexed in PyMuPDF
            
            # Draw rectangles over titles

            # Iterate over each title entry
            for index, row in material_title_df.iterrows():
                page_num = row['page']  # Get the page number for this title
                page = document[page_num]  # Pages are 0-indexed in PyMuPDF
                
                # Define the coordinates for the rectangle
                rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])
                
                # Draw the rectangle on the page
                page.draw_rect(rect, width=1, color=(1, 0, 0))
            
            # Draw rectangles over remarks on the correct page
            page_remarks = remarks_df[remarks_df['page'] == page_num]
            page = document[page_num - 1]
            for index, row in page_remarks.iterrows():
                # Define the coordinates for the rectangle
                rect = fitz.Rect(row['x0'], row['top'], row['x1'], row['bottom'])
                
                # Draw the rectangle on the page
                page.draw_rect(rect, width=1, color=(0, 1, 0))  # Green rectangle for remarks, width=1

    # Save the modified PDF to a new file
    document.save(output_pdf_path)
    document.close()

# print('horay')
