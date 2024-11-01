import cv2
import pytesseract
import numpy as np
import pandas as pd
import pickle
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from PIL import Image
from scipy.optimize import minimize
from django.conf import settings
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.conf import settings  # Import Django settings to access BASE_DIR

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#--------------------------------------------------- PRE TREATEMENT ---------------------------------------------------

# Function to convert all invoices to jpg extension with the same size
def convert_and_resize_images(input_folder, output_folder, size=(1700, 2338)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
        try:
            with Image.open(input_path) as img:
                img = img.convert('RGB')  # Convert to RGB format
                img = img.resize(size, Image.LANCZOS)  # Resize image with high-quality downsampling filter
                img.save(output_path, format='JPEG', quality=95)  # Save as JPEG
                print(f"Converted and resized {input_path} to {output_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

# Function to keep all black-and-white spectrum colors with all nuances
def keep_black_and_white(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Convert to grayscale if the image is in color
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Create a mask for all grayscale values (black to white spectrum)
    mask = np.full_like(img_gray, 255)
    mask = img_gray

    # Prepare the output image
    output_img = np.full_like(img_gray, 255)
    output_img[mask >= 0] = mask[mask >= 0]

    # Save the output image
    cv2.imwrite(output_path, output_img)
    return output_path

#--------------------------------------------- Bounding Boxes Detection ------------------------------------------------

# Function to detect red boxes in the image
def detect_red_boxes(image_path, target_image_path, output_image_path):
    print(f"Reading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    print("Image loaded successfully.")

    print(f"Reading target image from: {target_image_path}")
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        raise FileNotFoundError(f"Target image file not found: {target_image_path}")
    print("Target image loaded successfully.")

    # Get the DPI of the images
    img_dpi = Image.open(image_path).info.get('dpi', (96, 96))  # Default to 96 DPI if not found
    target_img_dpi = Image.open(target_image_path).info.get('dpi', (96, 96))

    # Calculate pixel per cm based on the DPI
    pixel_per_cm_img = img_dpi[0] / 254
    pixel_per_cm_target = target_img_dpi[0] / 254
    pixel_per_cm = (pixel_per_cm_img + pixel_per_cm_target) / 2  # Average DPI for better results
    print(f"Calculated pixel density: {pixel_per_cm:.2f} pixels per cm")

    print("Converting image to HSV color space...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("Image converted to HSV color space.")

    print("Creating mask for red color detection...")
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    mask = mask1 + mask2
    print("Red color mask created.")

    print("Finding contours in the mask...")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    print("Filtering for rectangular boxes...")
    boxes = [cv2.boundingRect(contour) for contour in contours if len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4]
    boxes.sort(key=lambda b: (b[1], b[0]))
    print(f"Detected {len(boxes)} rectangular boxes.")

    # Read the logo image for detection
    logo_path = 'media/logo.jpg'
    print(f"Reading logo from: {logo_path}")
    logo_img = cv2.imread(logo_path)
    if logo_img is None:
        raise FileNotFoundError(f"Logo file not found: {logo_path}")
    print("Logo image loaded successfully.")

    # Template matching to find the logo in the target image
    result = cv2.matchTemplate(target_img, logo_img, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # Adjust the threshold for detection
    loc = np.where(result >= threshold)

    if loc[0].size > 0:
        print("Logo detected in the target image.")
        # Get the first detected location of the logo
        top_left = (loc[1][0], loc[0][0])  # x, y coordinates of the logo
        logo_height = logo_img.shape[0]
        # Calculate the bottom edge of the logo
        logo_bottom_edge = top_left[1] + logo_height
        
        # Calculate the offsets for 1 cm and 2 cm based on pixel density
        vertical_offset = int(2 * pixel_per_cm)  # 1 cm offset in pixels
        horizontal_offset = int(40 * pixel_per_cm)  # 2 cm offset in pixels

        # New Y position for the first bounding box
        new_y_position = logo_bottom_edge + vertical_offset

    else:
        print("Logo not detected in the target image.")
        new_y_position = 0  # No offset if logo is not found

    # Before applying the boxes, optimize their positions
    optimized_boxes = optimize_boxes(target_img, boxes)

    # Adjust only the first bounding box and shift subsequent ones
    if new_y_position > 0 and optimized_boxes:
        first_box = optimized_boxes[0]
        optimized_boxes[0] = (first_box[0] + horizontal_offset, new_y_position, first_box[2], first_box[3])

        # Shift remaining boxes down by the same vertical offset and right by the horizontal offset while maintaining their relative spacing
        for i in range(1, len(optimized_boxes)):
            x, y, w, h = optimized_boxes[i]
            optimized_boxes[i] = (x + horizontal_offset, y + vertical_offset, w, h)

    save_optimized_boxes(optimized_boxes)

    print("Drawing detected boxes on the target image...")
    for i, (x, y, w, h) in enumerate(optimized_boxes):
        cv2.rectangle(target_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(target_img, f"Box {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    print("Detected boxes drawn on the target image.")

    print(f"Saving the result to: {output_image_path}")
    cv2.imwrite(output_image_path, target_img)
    print("Result image saved.")

    print("Process complete.")
    return optimized_boxes


#--------------------------------------------- Boxes Coordinates to CSV ------------------------------------------------

# Function to optimize box positions
def optimize_boxes(target_img, boxes):
    def objective_function(params, box, target_img):
        scale, tx, ty = params
        x, y, w, h = box
        x_new = int(scale * x + tx)
        y_new = int(scale * y + ty)
        w_new = int(scale * w)
        h_new = int(scale * h)
        roi = target_img[y_new:y_new + h_new, x_new:x_new + w_new]
        if roi.size == 0:
            return np.inf
        return np.sum(cv2.absdiff(roi, np.zeros_like(roi)))

    optimized_boxes = []
    for box in boxes:
        initial_params = [1, 0, 0]
        bounds = [(0.8, 1.2), (-50, 50), (-50, 50)]
        result = minimize(objective_function, initial_params, args=(box, target_img), bounds=bounds)
        scale, tx, ty = result.x
        x, y, w, h = box
        x_new = int(scale * x + tx)
        y_new = int(scale * y + ty)
        w_new = int(scale * w)
        h_new = int(scale * h)
        optimized_boxes.append((x_new, y_new, w_new, h_new))
    return optimized_boxes

# Function to save optimized boxes to CSV
def save_optimized_boxes(boxes):
    boxes_df = pd.DataFrame(boxes, columns=['X', 'Y', 'Width', 'Height'])
    output_path = os.path.join(settings.BASE_DIR, 'Extracted Data/Coordinates/optimized_red_boxes_coordinates.csv')
    boxes_df.to_csv(output_path, index=False)
    print(f"Optimized box coordinates saved to '{output_path}'.")

# Function to save optimized boxes to CSV and as an image filter
def save_optimized_boxes_and_filter_image(boxes, target_image_path):
    boxes_df = pd.DataFrame(boxes, columns=['X', 'Y', 'Width', 'Height'])
    output_csv_path = os.path.join(settings.BASE_DIR, 'Extracted Data/Coordinates/optimized_red_boxes_coordinates.csv')
    boxes_df.to_csv(output_csv_path, index=False)
    print(f"Optimized box coordinates saved to '{output_csv_path}'.")

    # Create a blank image with the same size as the target image
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        raise FileNotFoundError(f"Target image file not found: {target_image_path}")

    filter_img = np.zeros_like(target_img)
    for (x, y, w, h) in boxes:
        cv2.rectangle(filter_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red boxes on the filter image

    filter_image_path = os.path.join(settings.BASE_DIR, 'Extracted Data/Coordinates/filter.jpg')
    cv2.imwrite(filter_image_path, filter_img)
    print(f"Filter image with bounding boxes saved to '{filter_image_path}'.")

    return output_csv_path, filter_image_path
    
#---------------------------------------------- DATA PREPROCESSING -----------------------------------------------------

# Function to categorize text smartly
def smart_categorize_text(text):
    text = re.sub(r'[Oo]', '0', text.strip().replace('\n', '|'))
    text = re.sub(r'[Il]', '1', text)
    return text

def process_extracted_text(raw_text):
    # Detect dates with slashes in `dd/mm/yyyy` format and preserve them
    lines = [
        re.sub(r'[^0-9a-zA-Z|/. ]', '', line.strip())  # Keep slashes for dates
        if re.search(r'\b\d{2}/\d{2}/\d{4}\b', line)  # Preserve only if it matches date format
        else re.sub(r'[^0-9a-zA-Z|,. ]', '', line.strip())  # Regular cleaning for non-date lines
        or 'NAN' 
        for line in raw_text.strip().split('\n')
    ]
    
    # Removing redundant NANs
    final_lines = [lines[0]] + [
        lines[i] for i in range(1, len(lines)) if not (lines[i] == 'NAN' and lines[i-1] == 'NAN')
    ]
    
    return '|'.join(final_lines)

# Function to adequately split text
def clean_and_split_text(cell_text):
    cell_text = re.sub(r'\|{2,}', '|', cell_text)
    cell_text = cell_text.lstrip('|')
    return cell_text.split('|')

#----------------------------------------------- Filter Application --------------------------------------------------------------

# Function to apply the filter and extract text from the image
def apply_filter_and_extract_text(filtered_image_path, boxes_csv_path, output_csv_path):
    filtered_img = cv2.imread(filtered_image_path)
    if filtered_img is None:
        raise FileNotFoundError(f"Image file not found: {filtered_image_path}")
    
    boxes_df = pd.read_csv(boxes_csv_path)
    
    data = []
    for i, (x, y, w, h) in boxes_df.iterrows():
        roi = filtered_img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        
        raw_text = pytesseract.image_to_string(roi, config='--psm 6')
        processed_text = smart_categorize_text(process_extracted_text(raw_text))
        cell_type = 'Numeric' if any(c.isdigit() for c in processed_text) else 'Text' if any(c.isalpha() for c in processed_text) else 'Other'

        data.append({'Box Number': i + 1, 'Text': processed_text, 'Type': cell_type})
    
    # Save the extracted data to the CSV file
    pd.DataFrame(data).to_csv(output_csv_path, index=False)
    print(f"Extracted text data saved to: {output_csv_path}")

    # Return the output paths (both CSV and image filter paths)
    return output_csv_path


# Function to map the extracted data to the required schema and save it to dataset.csv
def map_data_and_save_to_csv(detected_csv_path, output_csv_path):
    mapping = {
        1: ['District'], 2: ['Num. Facture'], 3: ['Mois'], 4: ['Consommateur'],
        5: ['Adresse'], 6: ['Energie Enregistree (Phase 1)', 'Energie Enregistree (Phase 2)', 'Energie Enregistree (Phase 3)'],
        7: ['Index Ancien (Phase 1)', 'Index Ancien (Phase 2)', 'Index Ancien (Phase 3)'], 8: ['Coefficient multiplicateur (Phase 1)', 'Coefficient multiplicateur (Phase 2)', 'Coefficient multiplicateur (Phase 3)'],
        9: ['Index Nouveau (Phase 1)', 'Index Nouveau (Phase 2)', 'Index Nouveau (Phase 3)'], 10: ['Energie Enregistree (Jour)', 'Energie Enregistree (Pointe)', 'Energie Enregistree (Soir)', 'Energie Enregistree (Nuit)'],
        11: ['Index Ancien (Jour)', 'Index Ancien (Pointe)', 'Index Ancien (Soir)', 'Index Ancien (Nuit)'], 12: ['Index Nouveau (Jour)', 'Index Nouveau (Pointe)', 'Index Nouveau (Soir)', 'Index Nouveau (Nuit)'],
        13: ['Energie Enregistree (Reactif)'], 14: ['Index Nouveau (Reactif)'], 15: ['Index Ancien (Reactif)'], 16: ['Montant (Jour)', 'Montant (Pointe)', 'Montant (Soir)', 'Montant (Nuit)'],
        17: ['Prix Unitaire (Jour)', 'Prix Unitaire (Pointe)', 'Prix Unitaire (Soir)', 'Prix Unitaire (Nuit)'], 18: ['Consommation (Jour)', 'Consommation (Pointe)', 'Consommation (Soir)', 'Consommation (Nuit)'],
        19: ['Pointe Maximale hiver'], 20: ['Pointe Maximale Jour'], 21: ['Pointe Maximale Soir'], 22: ['Pointe Maximale Ete'],
        23: ['Montant sous Total'], 24: ['Consommation sous Total'], 25: ['Montant de Bonification'], 26: ['Contribution GMG'],
        27: ['Cos. Fi'], 28: ['Coef. K'], 29: ['Depassement Puissance'], 30: ['Avance/Consom.'], 31: ['Net a payer'], 32: ['Date limite du Payement']
    }
    
    print(f"Loading detected boxes from: {detected_csv_path}")
    df = pd.read_csv(detected_csv_path)
    print(f"Loaded {len(df)} rows from the detected CSV.")
    
    dataset = {field: [] for fields in mapping.values() for field in fields}
    
    print("Mapping data to the required schema...")
    for index, row in df.iterrows():
        box_number = row['Box Number']
        text = row['Text']
        
        if box_number in mapping:
            fields = mapping[box_number]
            values = clean_and_split_text(text)
            for i, field in enumerate(fields):
                dataset[field].append(values[i] if i < len(values) else 'NAN')
    
    dataset_df = pd.DataFrame({k: pd.Series(v) for k, v in dataset.items()})
    print("Data mapping complete.")
    
    # Add row number column
    dataset_df.insert(0, 'Row Number', dataset_df.index + 1)
    print("Added Row Number column.")
    
    print(f"Saving the mapped data to: {output_csv_path}")
    dataset_df.to_csv(output_csv_path, index=False)
    print("Data saved successfully.")







def pre_cleaning(df):
    # 1. Identify columns with specific data types based on column name keywords
    for col in df.columns:
        # Ensure values are treated as strings
        df[col] = df[col].astype(str)

        # For monetary and index-related columns
        if any(keyword in col.lower() for keyword in ['montant', 'prix', 'consommation', 'energie', 'index', 'pointe', 'net']):
            # Remove spaces and non-numeric characters except '.' and ','
            df[col] = df[col].apply(lambda x: re.sub(r'[^\d.,]', '', x))
            df[col] = df[col].replace(['', '.', ','], np.nan)  # Replace empty, '.', ',' with NaN
            
            # Ensure the column is treated as a string before using .str methods
            df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors

        # For date columns
        elif 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%d/%m/%Y')

        # For month columns, correctly parse and convert to datetime
        elif 'mois' in col.lower():
            df[col] = df[col].apply(lambda x: f"{x[:-4]}-01-{x[-4:]}" if len(x) == 6 else x)  # Convert to 'm-yyyy' format
            df[col] = pd.to_datetime(df[col], format='%m-%d-%Y', errors='coerce')  # Coerce to datetime

        # Clean all non-numeric columns (like 'Consommateur' and 'Adresse')
        else:
            df[col] = df[col].apply(lambda x: re.sub(r'[^\d./]', '', x))

    # 2. Fill NaNs with the previous row's value and then the next row's value if still NaN
    df.fillna(method='ffill', inplace=True)  # Forward fill
    df.fillna(method='bfill', inplace=True)  # Backward fill in case of leading NaNs

    # 3. Drop any empty columns
    df.dropna(axis=1, how='all', inplace=True)  # Drop columns that are completely empty

    # 4. Check for categorical columns for one-hot encoding only if needed
    categorical_cols = df.select_dtypes(include=['object']).columns.difference(['date', 'mois'])
    
    # Only apply one-hot encoding if there are categorical columns and they contain non-numeric data
    for cat_col in categorical_cols:
        if df[cat_col].nunique() > 10:  # Assuming >10 unique values might be more appropriate for numerical columns
            # Convert to numeric directly
            df[cat_col] = pd.to_numeric(df[cat_col], errors='coerce')
        else:
            df = pd.get_dummies(df, columns=[cat_col], drop_first=True)

    return df

def clean_dataset(df):
    # 1. Identify columns with specific data types based on column name keywords
    for col in df.columns:
        # Ensure values are treated as strings
        df[col] = df[col].astype(str)

        # For monetary and index-related columns
        if any(keyword in col.lower() for keyword in ['montant', 'prix', 'consommation', 'energie', 'index', 'pointe', 'net']):
            # Remove spaces and non-numeric characters except '.' and ','
            df[col] = df[col].apply(lambda x: re.sub(r'[^\d.,]', '', x))
            df[col] = df[col].replace(['', '.', ','], np.nan)  # Replace empty, '.', ',' with NaN
            
            # Replace commas with dots and convert to numeric
            df[col] = df[col].str.replace(',', '.').astype(float)

        # For date columns
        elif 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%d/%m/%Y')

        # For month columns, correctly parse and convert to datetime
        elif 'mois' in col.lower():
            df[col] = df[col].apply(lambda x: f"{x[:-4]}-01-{x[-4:]}" if len(x) == 6 else x)  # Convert to 'm-yyyy' format
            df[col] = pd.to_datetime(df[col], format='%m-%d-%Y', errors='coerce')  # Coerce to datetime

        # Clean all non-numeric columns (like 'Consommateur' and 'Adresse')
        else:
            df[col] = df[col].apply(lambda x: re.sub(r'[^\d./]', '', x))
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric if possible

    # 2. Fill NaNs with the previous row's value and then the next row's value if still NaN
    df.ffill(inplace=True)  # Forward fill
    df.bfill(inplace=True)  # Backward fill in case of leading NaNs

    # 3. Drop any empty columns
    df.dropna(axis=1, how='all', inplace=True)  # Drop columns that are completely empty

    # 4. Check for categorical columns for one-hot encoding only if needed
    categorical_cols = df.select_dtypes(include=['object']).columns.difference(['date', 'mois'])
    
    # Only apply one-hot encoding if there are categorical columns and they contain non-numeric data
    for cat_col in categorical_cols:
        if df[cat_col].nunique() > 10:  # Assuming >10 unique values might be more appropriate for numerical columns
            # Convert to numeric directly
            df[cat_col] = pd.to_numeric(df[cat_col], errors='coerce')
        else:
            df = pd.get_dummies(df, columns=[cat_col], drop_first=True)

    return df

def augment_data(df, n_augmented=1000):
    """Augment data by generating random values."""
    augmented_data = []
    for _ in range(n_augmented):
        random_row = df.sample(n=1).values.flatten()
        noise = np.random.normal(0, 0.1, size=random_row.shape)  # Adjust the noise level as necessary
        augmented_row = [value + value * noise[i] if isinstance(value, (int, float)) else value 
                         for i, value in enumerate(random_row)]
        augmented_data.append(augmented_row)
    
    # Create a DataFrame from the augmented data
    return pd.concat([df, pd.DataFrame(augmented_data, columns=df.columns)], ignore_index=True)

def predict_from_date(input_date):
    """Cleans the dataset, augments the data, and predicts the invoice value for a future date."""
    
    # Load dataset
    df = pd.read_csv('Extracted Data/dataset.csv')
    df_cleaned = clean_dataset(pre_cleaning(df))

    # Set target variable and features
    target_variable = 'Net a payer'
    X = df_cleaned.drop(columns=[target_variable])
    y = df_cleaned[target_variable]

    # Data Augmentation
    df_augmented = augment_data(df_cleaned)

    # Update features and target variable after augmentation
    X_augmented = df_augmented.drop(columns=[target_variable])
    y_augmented = df_augmented[target_variable]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

    # Model setup and training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prepare features for prediction
    future_date = pd.to_datetime(input_date, format='%d/%m/%Y')
    future_features = {col: [0] for col in X_augmented.columns}  # Initialize with zeros
    future_features['date'] = [future_date.toordinal()]  # Assuming 'date' is a feature in ordinal format

    future_df = pd.DataFrame(future_features).reindex(columns=X_augmented.columns, fill_value=X_augmented.mean())

    # Predict the future invoice value
    predicted_value = model.predict(future_df)

    return predicted_value[0]




# Function to run all the utils functions in order
def run_utils():
    # Directory paths
    base_dir = os.getcwd()  # Assuming you're using the current working directory as the base
    input_folder_path = os.path.join(base_dir, 'media/Inputs')
    processed_output_folder = os.path.join(base_dir, 'media/Processed')
    filtered_output_folder = os.path.join(base_dir, 'media/Filtered')
    detected_output_folder = os.path.join(base_dir, 'media/Detected')
    detected_coordinates_path = os.path.join(base_dir, 'Extracted Data/Detected')
    dataset_csv_output_path = os.path.join(base_dir, 'Extracted Data/dataset.csv')

    # Create directories if they do not exist
    os.makedirs(filtered_output_folder, exist_ok=True)
    os.makedirs(detected_output_folder, exist_ok=True)
    os.makedirs(processed_output_folder, exist_ok=True)

    # Initialize the list to store results
    all_detected_data = []

    # Step 1: Convert and resize input images
    convert_and_resize_images(input_folder_path, processed_output_folder)

    # Step 2: Iterate over all JPG files in the processed folder
    for filename in os.listdir(processed_output_folder):
        if filename.lower().endswith('.jpg'):
            input_image_path = os.path.join(processed_output_folder, filename)
            filtered_image_path = os.path.join(filtered_output_folder, filename)
            detected_image_path = os.path.join(detected_output_folder, filename.replace('.jpg', '_detected.jpg'))
            detected_csv_path = os.path.join(detected_output_folder, filename.replace('.jpg', '_detected_data.csv'))
            optimized_boxes_csv_path = os.path.join(base_dir, 'Extracted Data/Coordinates/optimized_red_boxes_coordinates.csv')

            # Step 3: Apply black and white filter
            keep_black_and_white(input_image_path, filtered_image_path)
            
            # Step 4: Detect red boxes
            detect_red_boxes(os.path.join(base_dir, 'Selection/manual_selection_final.png'), filtered_image_path, detected_image_path)
            
            # Step 5: Generate and save optimized boxes and filter image
            optimized_boxes_path, _ = save_optimized_boxes_and_filter_image(
                pd.read_csv(optimized_boxes_csv_path).values.tolist(), 
                filtered_image_path
            )

            # Step 6: Apply filter and extract text
            apply_filter_and_extract_text(filtered_image_path, optimized_boxes_path, detected_csv_path)
            
            # Step 7: Read the detected data and append to the list
            detected_data = pd.read_csv(detected_csv_path)
            all_detected_data.append(detected_data)

    # Step 8: Concatenate all detected data and save to all_detected_data.csv
    if all_detected_data:
        all_detected_df = pd.concat(all_detected_data, ignore_index=True)
        all_detected_df.to_csv(os.path.join(base_dir, 'Extracted Data/all_detected_data.csv'), index=False)

        # Step 9: Map data and save to final dataset.csv
        map_data_and_save_to_csv(os.path.join(base_dir, 'Extracted Data/all_detected_data.csv'), dataset_csv_output_path)
    
    print(f"Processing completed. Final dataset saved at: {dataset_csv_output_path}")

