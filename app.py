import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.embeddings import Embeddings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import warnings
warnings.filterwarnings('ignore')

# Load Gemini API Key from environment, secrets, or .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
if not GEMINI_API_KEY and os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    GEMINI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
                elif line.startswith("GOOGLE_API_KEY="):
                    GEMINI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except Exception:
        pass

if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


class GeminiEmbeddingsWrapper(Embeddings):
    def __init__(self, gemini_api_key):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=gemini_api_key)
        
    def embed_documents(self, texts):
        return [self.embeddings.embed_query(text) for text in texts]
        
    def embed_query(self, text):
        return self.embeddings.embed_query(text)





def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Resume Analyzer AI', layout="wide")

    # Injecting rich, custom CSS styles for maximum visual wow factor and premium glassmorphic feel
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Font & App View Style */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: #0b0c10 !important;
        color: #c5c6c7 !important;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #1f2833 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Header customization */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Stunning Glassmorphic Cards */
    .stForm, div[data-testid="stForm"] {
        background: rgba(31, 40, 51, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(8px) !important;
        margin-bottom: 25px !important;
    }

    /* Card titles and Headers */
    .card-title {
        font-size: 24px;
        font-weight: 700;
        color: #45f3ff;
        margin-top: 20px;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
        text-shadow: 0 0 10px rgba(69, 243, 255, 0.2);
    }

    /* Premium Result Box */
    .result-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border-left: 5px solid #ff007f !important;
        border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 4px 12px 12px 4px !important;
        padding: 24px !important;
        box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 30px !important;
        line-height: 1.6 !important;
        color: #e5e5e5 !important;
        font-size: 16px !important;
    }

    /* Premium Job Card Styling */
    .job-card {
        background: rgba(31, 40, 51, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-left: 5px solid #45f3ff !important;
        border-radius: 6px 16px 16px 6px !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 8px 30px 0 rgba(0,0,0,0.3) !important;
    }

    .job-header {
        font-size: 20px;
        font-weight: 600;
        color: #45f3ff;
        margin-bottom: 12px;
    }

    .job-field {
        font-size: 15px;
        margin-bottom: 6px;
        color: #d1d2d3;
    }

    .job-link {
        color: #ff007f !important;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    .job-link:hover {
        color: #ff55aa !important;
        text-shadow: 0 0 8px rgba(255, 0, 127, 0.4);
    }

    /* Button Custom Design */
    div.stButton > button {
        background: linear-gradient(135deg, #ff007f, #7f00ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 30px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.3) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 0, 127, 0.5) !important;
        background: linear-gradient(135deg, #ff1a8c, #8c1aff) !important;
    }

    div.stButton > button:active {
        transform: translateY(0.5px) !important;
    }

    /* Text Inputs Custom Design */
    input {
        background-color: #0b0c10 !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }

    input:focus {
        border-color: #45f3ff !important;
        box-shadow: 0 0 10px rgba(69,243,255,0.3) !important;
    }

    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 2px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #45f3ff !important;
        background: rgba(69, 243, 255, 0.02) !important;
    }

    /* Main title styling */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #45f3ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-shadow: 0 0 25px rgba(69, 243, 255, 0.15);
    }

    .sub-title {
        text-align: center;
        font-size: 16px;
        color: #8b9bb4;
        margin-bottom: 40px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Top Brand Header
    st.markdown('<div class="main-title">Resume Analyzer AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Powering your job hunt with Gemini 2.5 Flash</div>', unsafe_allow_html=True)


class resume_analyzer:

    def pdf_to_chunks(pdf):
        # read pdf and it returns memory address
        pdf_reader = PdfReader(pdf)

        # extrat text from each page separately
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the long text into small chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)

        chunks = text_splitter.split_text(text=text)
        return chunks


    def gemini(gemini_api_key, chunks, analyze):

        # Using Google Gemini service for embedding (with custom wrapper to prevent batching errors)
        embeddings = GeminiEmbeddingsWrapper(gemini_api_key=gemini_api_key)

        # Facebook AI Similarity Search library help us to convert text data to numerical vector
        vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

        # compares the query and chunks, enabling the selection of the top 'K' most similar chunks based on their similarity scores.
        docs = vectorstores.similarity_search(query=analyze, k=3)

        # creates a Gemini object, using the Gemini 2.5 Flash model
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=gemini_api_key)

        # question-answering (QA) pipeline, making use of the load_qa_chain function
        chain = load_qa_chain(llm=llm, chain_type='stuff')

        response = chain.run(input_documents=docs, question=analyze)
        return response


    def summary_prompt(query_with_chunks):

        query = f''' need to detailed summarization of below resume and finally conclude them

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_summary():

        with st.form(key='Summary'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        add_vertical_space(3)
        if submit:
            if pdf is not None and GEMINI_API_KEY != '':
                try:
                    with st.spinner('Processing...'):

                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.gemini(gemini_api_key=GEMINI_API_KEY, chunks=pdf_chunks, analyze=summary_prompt)

                    st.markdown(f'''
                    <div class="card-title">✨ Resume Summary</div>
                    <div class="result-card">
                        {summary}
                    </div>
                    ''', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif GEMINI_API_KEY == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Set Your Gemini API Key in the .env file</h5>', unsafe_allow_html=True)


    def strength_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the strength of below resume and finally conclude them
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_strength():

        with st.form(key='Strength'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)

        add_vertical_space(3)
        if submit:
            if pdf is not None and GEMINI_API_KEY != '':
                try:
                    with st.spinner('Processing...'):
                    
                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.gemini(gemini_api_key=GEMINI_API_KEY, chunks=pdf_chunks, analyze=summary_prompt)
                        
                        strength_prompt = resume_analyzer.strength_prompt(query_with_chunks=summary)

                        strength = resume_analyzer.gemini(gemini_api_key=GEMINI_API_KEY, chunks=pdf_chunks, analyze=strength_prompt)

                    st.markdown(f'''
                    <div class="card-title">💪 Key Strengths</div>
                    <div class="result-card">
                        {strength}
                    </div>
                    ''', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif GEMINI_API_KEY == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Set Your Gemini API Key in the .env file</h5>', unsafe_allow_html=True)


    def weakness_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the weakness of below resume and how to improve make a better resume.

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_weakness():

        with st.form(key='Weakness'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        add_vertical_space(3)
        if submit:
            if pdf is not None and GEMINI_API_KEY != '':
                try:
                    with st.spinner('Processing...'):
                    
                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.gemini(gemini_api_key=GEMINI_API_KEY, chunks=pdf_chunks, analyze=summary_prompt)

                        weakness_prompt = resume_analyzer.weakness_prompt(query_with_chunks=summary)

                        weakness = resume_analyzer.gemini(gemini_api_key=GEMINI_API_KEY, chunks=pdf_chunks, analyze=weakness_prompt)

                    st.markdown(f'''
                    <div class="card-title">🛠️ Weaknesses & Improvements</div>
                    <div class="result-card">
                        {weakness}
                    </div>
                    ''', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif GEMINI_API_KEY == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Set Your Gemini API Key in the .env file</h5>', unsafe_allow_html=True)


    def job_title_prompt(query_with_chunks):

        query = f''' what are the job roles i apply to likedin based on below?
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def job_title_suggestion():

        with st.form(key='Job Titles'):

            # User Upload the Resume
            add_vertical_space(1)
            pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
            add_vertical_space(1)

            # Click on Submit Button
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)

        add_vertical_space(3)
        if submit:
            if pdf is not None and GEMINI_API_KEY != '':
                try:
                    with st.spinner('Processing...'):
                    
                        pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

                        summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)

                        summary = resume_analyzer.gemini(gemini_api_key=GEMINI_API_KEY, chunks=pdf_chunks, analyze=summary_prompt)

                        job_title_prompt = resume_analyzer.job_title_prompt(query_with_chunks=summary)

                        job_title = resume_analyzer.gemini(gemini_api_key=GEMINI_API_KEY, chunks=pdf_chunks, analyze=job_title_prompt)

                    st.markdown(f'''
                    <div class="card-title">🎯 Recommended Job Roles</div>
                    <div class="result-card">
                        {job_title}
                    </div>
                    ''', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)

            elif pdf is None:
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Upload Your Resume</h5>', unsafe_allow_html=True)
            
            elif GEMINI_API_KEY == '':
                st.markdown(f'<h5 style="text-align: center;color: orange;">Please Set Your Gemini API Key in the .env file</h5>', unsafe_allow_html=True)



class linkedin_scraper:

    def webdriver_setup():
            
        options = webdriver.ChromeOptions()
        options.binary_location = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        return driver


    def get_userinput():

        add_vertical_space(2)
        with st.form(key='linkedin_scarp'):

            add_vertical_space(1)
            col1,col2,col3 = st.columns([0.5,0.3,0.2], gap='medium')
            with col1:
                job_title_input = st.text_input(label='Job Title')
                job_title_input = job_title_input.split(',')
            with col2:
                job_location = st.text_input(label='Job Location', value='India')
            with col3:
                job_count = st.number_input(label='Job Count', min_value=1, value=1, step=1)

            # Submit Button
            add_vertical_space(1)
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        
        return job_title_input, job_location, job_count, submit


    def build_url(job_title, job_location):

        b = []
        for i in job_title:
            x = i.split()
            y = '%20'.join(x)
            b.append(y)

        job_title = '%2C%20'.join(b)
        link = f"https://in.linkedin.com/jobs/search?keywords={job_title}&location={job_location}&locationId=&geoId=102713980&f_TPR=r604800&position=1&pageNum=0"

        return link
    

    def open_link(driver, link):

        while True:
            # Break the Loop if the Element is Found, Indicating the Page Loaded Correctly
            try:
                driver.get(link)
                driver.implicitly_wait(5)
                time.sleep(3)
                driver.find_element(by=By.CSS_SELECTOR, value='span.switcher-tabs__placeholder-text.m-auto')
                return
            
            # Retry Loading the Page
            except NoSuchElementException:
                continue


    def link_open_scrolldown(driver, link, job_count):
        
        # Open the Link in LinkedIn
        linkedin_scraper.open_link(driver, link)

        # Scroll Down the Page
        for i in range(0,job_count):

            # Simulate clicking the Page Up button
            body = driver.find_element(by=By.TAG_NAME, value='body')
            body.send_keys(Keys.PAGE_UP)

            # Locate the sign-in modal dialog 
            try:
                driver.find_element(by=By.CSS_SELECTOR, 
                                value="button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']>icon>svg").click()
            except:
                pass

            # Scoll down the Page to End
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.implicitly_wait(2)

            # Click on See More Jobs Button if Present
            try:
                x = driver.find_element(by=By.CSS_SELECTOR, value="button[aria-label='See more jobs']").click()
                driver.implicitly_wait(5)
            except:
                pass


    def job_title_filter(scrap_job_title, user_job_title_input):
        
        # User Job Title Convert into Lower Case
        user_input = [i.lower().strip() for i in user_job_title_input]

        # scraped Job Title Convert into Lower Case
        scrap_title = [i.lower().strip() for i in [scrap_job_title]]

        # Verify Any User Job Title in the scraped Job Title
        confirmation_count = 0
        for i in user_input:
            if all(j in scrap_title[0] for j in i.split()):
                confirmation_count += 1

        # Return Job Title if confirmation_count greater than 0 else return NaN
        if confirmation_count > 0:
            return scrap_job_title
        else:
            return np.nan


    def scrap_company_data(driver, job_title_input, job_location):

        # scraping the Company Data
        company = driver.find_elements(by=By.CSS_SELECTOR, value='h4[class="base-search-card__subtitle"]')
        company_name = [i.text for i in company]

        location = driver.find_elements(by=By.CSS_SELECTOR, value='span[class="job-search-card__location"]')
        company_location = [i.text for i in location]

        title = driver.find_elements(by=By.CSS_SELECTOR, value='h3[class="base-search-card__title"]')
        job_title = [i.text for i in title]

        url = driver.find_elements(by=By.XPATH, value='//a[contains(@href, "/jobs/")]')
        website_url = [i.get_attribute('href') for i in url]

        # combine the all data to single dataframe
        df = pd.DataFrame(company_name, columns=['Company Name'])
        df['Job Title'] = pd.DataFrame(job_title)
        df['Location'] = pd.DataFrame(company_location)
        df['Website URL'] = pd.DataFrame(website_url)

        # Return Job Title if there are more than 1 matched word else return NaN
        df['Job Title'] = df['Job Title'].apply(lambda x: linkedin_scraper.job_title_filter(x, job_title_input))

        # Return Location if User Job Location in Scraped Location else return NaN
        df['Location'] = df['Location'].apply(lambda x: x if job_location.lower() in x.lower() else np.nan)
        
        # Drop Null Values and Reset Index
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)

        return df 
        

    def scrap_job_description(driver, df, job_count):
        
        # Get URL into List
        website_url = df['Website URL'].tolist()
        
        # Scrap the Job Description
        job_description = []
        description_count = 0

        for i in range(0, len(website_url)):
            try:
                # Open the Link in LinkedIn
                linkedin_scraper.open_link(driver, website_url[i])

                # Click on Show More Button
                driver.find_element(by=By.CSS_SELECTOR, value='button[data-tracking-control-name="public_jobs_show-more-html-btn"]').click()
                driver.implicitly_wait(5)
                time.sleep(1)

                # Get Job Description
                description = driver.find_elements(by=By.CSS_SELECTOR, value='div[class="show-more-less-html__markup relative overflow-hidden"]')
                data = [i.text for i in description][0]
                
                # Check Description length and Duplicate
                if len(data.strip()) > 0 and data not in job_description:
                    job_description.append(data)
                    description_count += 1
                else:
                    job_description.append('Description Not Available')
            
            # If any unexpected issue 
            except:
                job_description.append('Description Not Available')
            
            # Check Description Count reach User Job Count
            if description_count == job_count:
                break

        # Filter the Job Description
        df = df.iloc[:len(job_description), :]

        # Add Job Description in Dataframe
        df['Job Description'] = pd.DataFrame(job_description, columns=['Description'])
        df['Job Description'] = df['Job Description'].apply(lambda x: np.nan if x=='Description Not Available' else x)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df


    def display_data_userinterface(df_final):

        # Display the Data in User Interface
        add_vertical_space(1)
        if len(df_final) > 0:
            for i in range(0, len(df_final)):
                
                st.markdown(f'''
                <div class="job-card">
                    <div class="job-header">💼 Job Posting Details #{i+1}</div>
                    <div class="job-field"><strong>🏢 Company:</strong> {df_final.iloc[i,0]}</div>
                    <div class="job-field"><strong>🎯 Title:</strong> {df_final.iloc[i,1]}</div>
                    <div class="job-field"><strong>📍 Location:</strong> {df_final.iloc[i,2]}</div>
                    <div class="job-field"><strong>🌐 Link:</strong> <a href="{df_final.iloc[i,3]}" target="_blank" class="job-link">View on LinkedIn</a></div>
                </div>
                ''', unsafe_allow_html=True)
                with st.expander(label='📝 Job Description Details'):
                    st.write(df_final.iloc[i, 4])
                add_vertical_space(1)
        
        else:
            st.markdown(f'<h5 style="text-align: center;color: orange;">No Matching Jobs Found</h5>', 
                                unsafe_allow_html=True)


    def main():
        
        # Initially set driver to None
        driver = None
        
        try:
            job_title_input, job_location, job_count, submit = linkedin_scraper.get_userinput()
            add_vertical_space(2)
            
            if submit:
                if job_title_input != [] and job_location != '':
                    
                    with st.spinner('Chrome Webdriver Setup Initializing...'):
                        driver = linkedin_scraper.webdriver_setup()
                                       
                    with st.spinner('Loading More Job Listings...'):

                        # build URL based on User Job Title Input
                        link = linkedin_scraper.build_url(job_title_input, job_location)

                        # Open the Link in LinkedIn and Scroll Down the Page
                        linkedin_scraper.link_open_scrolldown(driver, link, job_count)

                    with st.spinner('scraping Job Details...'):

                        # Scraping the Company Name, Location, Job Title and URL Data
                        df = linkedin_scraper.scrap_company_data(driver, job_title_input, job_location)

                        # Scraping the Job Descriptin Data
                        df_final = linkedin_scraper. scrap_job_description(driver, df, job_count)
                    
                    # Display the Data in User Interface
                    linkedin_scraper.display_data_userinterface(df_final)

                
                # If User Click Submit Button and Job Title is Empty
                elif job_title_input == []:
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Title is Empty</h5>', 
                                unsafe_allow_html=True)
                
                elif job_location == '':
                    st.markdown(f'<h5 style="text-align: center;color: orange;">Job Location is Empty</h5>', 
                                unsafe_allow_html=True)

        except Exception as e:
            add_vertical_space(2)
            st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)
        
        finally:
            if driver:
                driver.quit()



# Streamlit Configuration Setup
streamlit_config()
add_vertical_space(2)



with st.sidebar:

    add_vertical_space(4)

    option = option_menu(menu_title='', options=['Summary', 'Strength', 'Weakness', 'Job Titles', 'Linkedin Jobs'],
                         icons=['house-fill', 'database-fill', 'pass-fill', 'list-ul', 'linkedin'])



if option == 'Summary':

    resume_analyzer.resume_summary()



elif option == 'Strength':

    resume_analyzer.resume_strength()



elif option == 'Weakness':

    resume_analyzer.resume_weakness()



elif option == 'Job Titles':

    resume_analyzer.job_title_suggestion()



elif option == 'Linkedin Jobs':
    
    linkedin_scraper.main()


