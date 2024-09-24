# Document Summarization App using Language Models

This project creates a simple PDF document summarization app using Streamlit and Hugging Face language models (LaMini-Flan-T5-248M). The app allows users to upload a PDF file and generate a summary using the LaMini-Flan-T5-248M model.

## Features
- **PDF Upload**: Upload any PDF document for processing.
- **Summarization**: Uses the LaMini-Flan-T5-248M language model to summarize the document.
- **Streamlit UI**: Simple and intuitive UI to upload and view the PDF alongside the summary.

## Project Setup

Follow these steps to replicate the project in your local environment.

### Step 1: Clone the Repository

Start by cloning the necessary model repository from Hugging Face:
```python
git lfs install  # Ensure git-lfs is installed
git clone https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M
```
If your network connection is slow, you might want to want to clone without large files (just their pointers). But keep in mind to manually download the skipped files:
```python
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M
```
Make sure that the LaMini-Flan-T5-248M directory is in the same folder as your project code.

### Step 2: Set up the Environment

Create and activate a Python virtual environment using Conda (not a must, but recommended):
```python
conda create -n myenv python=3.10
conda activate myenv
```

### Step 3: Install Required Packages

Install the necessary dependencies using conda and pip:

#### Install with Conda:
```python
conda install -c conda-forge langchain pytorch sentencepiece transformers accelerate pypdf
```

#### Install the remaining packages with Pip:
```python
pip install sentence-transformers chromadb tiktoken streamlit
```
### Step 4: Running the App

Once all dependencies are installed, you can run the Streamlit app. Make sure the cloned `LaMini-Flan-T5-248M` model directory is in the same directory as your Streamlit script.
```python
streamlit run streamlit_app.py
```
## Directory Structure

Your directory structure should look something like this:
```plaintext
├── LaMini-Flan-T5-248M
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model.bin
│   ├── spiece.model
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── other model files...
│
├── streamlit_app.py
├── data
│   └── (Uploaded PDF files will be saved here)
└── README.md
```
## Usage
- Upload PDF: When you open the app, you'll be prompted to upload a PDF file.
- Summarize: Once uploaded, click the "Summarize" button to generate a summary of the document.
- View Results: The app will display the PDF on one side and the summary on the other.

## Troubleshooting
- Memory Issues: If you're running into memory issues (especially if you're using a GPU), you might need to configure the model loading options with offload_folder or use safetensors. This has been accounted for in the model loading code, but adjust as needed based on your hardware.
- Offloading Model Weights: Ensure there is sufficient disk space if you're offloading parts of the model to disk.

## Future Improvements
- Add more language models for experimentation.
- Implement chunking strategies for large documents.
- Allow users to select specific sections of the document for summarization.

## Credits
- LangChain: For document loading and text splitting.
- Hugging Face: For the LaMini-Flan-T5-248M language model.
- Streamlit: For the front-end interface.
