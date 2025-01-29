import easyocr
import matplotlib.pyplot as plt
import cv2
import re
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

def process_image(image_file):
    try:
        import numpy as np
        
        # Initialize the EasyOCR reader
        reader = easyocr.Reader(['en'])  # Specify language(s)

        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Perform OCR
        result = reader.readtext(image)
        raw_text = " ".join([detection[1] for detection in result])

        # Define the LLM prompt
        template = """
        Extract the following details: invoice no., Description, Quantity, Date,
        Unit price, Amount, Total, email, phone number, and address from the text: {raw_text}

        Expected output (JSON format):
        {{
            'Invoice no.': '2001321',
            'Description': 'HP Laptop',
            'Quantity': '1',
            'Date': '5/4/2023',
            'Unit price': '500.00',
            'Amount': '500.00',
            'Total': '500.00',
            'Email': 'sharathkumarraju@proton.me',
            'Phone number': '8888888888',
            'Address': 'Hyderabad, India'
        }}
        """
        prompt = template.format(raw_text=raw_text)
        
        # Ensure the LLM is initialized
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment variables.")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # Get response from the LLM
        response = llm.predict(text=prompt, temperature=0.1)
        
        # Extract JSON from the LLM response
        json_match = re.search(r'{.*}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

    except Exception as e:
        print(f"Error processing image: {e}")
        return None