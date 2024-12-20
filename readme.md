A Text Summarization Application
=> Enter any text and get the summary of it.


Prerequisite
=> Python


Commands to setup Virtual Environment (in Windows) :-

to Install virtual environment => pip install virtualenv

to Create virtual environment => virtualenv <my_virtual_environment>

to Activate virtual environment => <my_virtual_environment>/Scripts/activate

to Install required packages in virtual environment => pip install -r ./requirements.txt

to Deactivate virtual environment => deactivate


Steps to get the trained model :-

=> Train the model on "https://www.kaggle.com/" using the code written in file "mode_training.ipynb"

=> Download the "saved_summary_model"

=> Decompress the folder

=> Add the folder to project directory


Command to run the project :-

=> uvicorn app:app --reload
