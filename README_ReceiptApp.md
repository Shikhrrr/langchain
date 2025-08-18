# 🧾 Receipt OCR Handler - Streamlit App

A modern, user-friendly Streamlit frontend for extracting and analyzing receipt data using OCR and LangChain.

## ✨ Features

- **📤 Image Upload**: Drag & drop or select receipt images (PNG, JPG, JPEG)
- **🔍 OCR Processing**: Extract text from receipt images using OCR.space API
- **🏗️ Structured Data Extraction**: Use LangChain to parse receipt text into structured data
- **💾 Database Storage**: Save extracted data to PostgreSQL database
- **❓ Smart Querying**: Ask questions about your receipt data using natural language
- **📊 Data Visualization**: Clean, tabular display of extracted information
- **🎨 Modern UI**: Beautiful, responsive interface with custom styling

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** installed
2. **PostgreSQL** database running
3. **Ollama** with llama3 model running locally
4. **OCR.space API key** (free tier available)

### Installation

1. **Clone or navigate to your project directory:**
   ```bash
   cd /Users/shikhar/Desktop/langchain
   ```

2. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your database:**
   - Ensure PostgreSQL is running
   - Create a database named `receipt_db`
   - Update the database URI in the app if needed

5. **Start Ollama with llama3 model:**
   ```bash
   ollama run llama3
   ```

### Running the App

```bash
streamlit run receipt_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🏗️ Architecture

### Components

1. **Frontend (Streamlit)**: User interface for image upload and interaction
2. **OCR Engine (OCR.space)**: Text extraction from images
3. **LangChain Pipeline**: Structured data extraction and query processing
4. **Database (PostgreSQL)**: Persistent storage of receipt data
5. **LLM (Ollama/llama3)**: Natural language understanding and reasoning

### Data Flow

```
Image Upload → OCR Processing → Text Extraction → LangChain Parsing → Database Storage → Query Processing → Results
```

## 📱 Usage Guide

### 1. Upload Receipt
- Click "Choose an image file" or drag & drop
- Supported formats: PNG, JPG, JPEG
- The image will be displayed for verification

### 2. Extract Text
- Click "🔍 Extract Text (OCR)" button
- Wait for OCR processing to complete
- Review the extracted text in the right panel

### 3. Extract Structured Data
- Click "🏗️ Extract Structured Data" button
- LangChain will parse the text into structured format
- View the extracted items in a clean table

### 4. Save to Database
- Click "💾 Save to Database" button
- Data will be stored in PostgreSQL
- Check sidebar for database status

### 5. Query Your Data
- Use example query buttons for common questions
- Or type your own custom query
- Get intelligent answers using SQL agents and reasoning

## 🔧 Configuration

### Database Connection
Update the database URI in the sidebar:
```
postgresql://username:password@localhost/database_name
```

### OCR API Key
The app uses a default OCR.space API key. For production use, consider:
- Getting your own API key from [OCR.space](https://ocr.space/ocrapi)
- Setting it as an environment variable
- Updating the `ocr_space_file` function

### Ollama Model
The app uses `llama3` by default. To use a different model:
- Update the model name in `setup_langchain()` function
- Ensure the model is available in Ollama

## 📊 Database Schema

```sql
CREATE TABLE receipt_items (
    id SERIAL PRIMARY KEY,
    item_name VARCHAR,
    quantity INTEGER,
    price DOUBLE PRECISION
);
```

## 🎯 Example Queries

### Factual Queries
- "Which item was sold the most?"
- "What's the total revenue?"
- "How many items cost more than ₹1500?"
- "List all items with quantity greater than 10"

### Subjective Queries
- "Recommend which items I should buy and which to avoid"
- "What are the best deals in my purchase history?"
- "Which items offer the best value for money?"

## 🛠️ Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check if PostgreSQL is running
   - Verify database credentials
   - Ensure database `receipt_db` exists

2. **LangChain Setup Failed**
   - Check if Ollama is running
   - Verify llama3 model is available
   - Check Python dependencies

3. **OCR Failed**
   - Verify internet connection
   - Check OCR.space API status
   - Ensure image format is supported

4. **Image Upload Issues**
   - Check file format (PNG, JPG, JPEG only)
   - Ensure file size is reasonable
   - Try a different image

### Debug Mode
To see detailed error messages, check the Streamlit console output.

## 🔒 Security Notes

- The app includes a hardcoded OCR API key (for demo purposes)
- In production, use environment variables for sensitive data
- Database credentials should be properly secured
- Consider implementing user authentication for multi-user scenarios

## 🚀 Future Enhancements

- **Multi-language Support**: OCR and processing in multiple languages
- **Receipt Categories**: Automatic categorization of receipts
- **Export Features**: CSV, PDF, Excel export options
- **Analytics Dashboard**: Charts and insights
- **Mobile Optimization**: Better mobile experience
- **Batch Processing**: Handle multiple receipts at once

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console output for error messages
3. Ensure all prerequisites are met
4. Verify database and Ollama connectivity

## 📄 License

This project is part of your personal workspace. Feel free to modify and extend as needed.

---

**Happy Receipt Processing! 🧾✨**
