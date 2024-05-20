# https://github.com/openai/openai-python
import os
import config
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
    api_key=config.APIKEY
)


def grade_essay(essay_text):
    sysprompt="""You are the best teacher in IELTS, you have to grade essay of student. The criterias in bellow.
    Task_Achievement: This assesses how well the student fulfills the requirements of the given task, including addressing all parts of the task, presenting a clear and focused response, and ensuring relevance to the essay.
    Coherence_and_Cohesion: This evaluates the clarity and organization of the writing, ensuring ideas are logically arranged and easy to follow. Cohesion involves using devices like linking words, pronouns, and conjunctions to connect ideas, sentences, and paragraphs
    Lexical_Resource_Vocabulary: This criterion focuses on the range and use of vocabulary in the writing, including the appropriateness, variety, and precision of word choice. Effective use of vocabulary enhances the quality of writing
    Grammatical_Range_and_Accuracy: This measures the writer's ability to use a range of grammatical structures accurately, assessing the ability to express ideas without errors that hinder communication. High grammatical control contributes to overall clarity and effectiveness of the writing
    Overall_grade: Overall assessment. Teachers' comments must be friendly to help students understand and be motivated to learn and strive to continue.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sysprompt,
            },
            {
                "role": "user",
                "content": essay_text,
            },
        ],
        model="gpt-4o",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "CEFR_test",
                    "description": "The test assesses your English proficiency in grammar, vocabulary, reading, writing and can be used for self-improvement, to prove your English level to an student or to prepare for a test such as IELTS, TOEFL",
                    "parameters": {
                        "type": "object",
                        "properties": {                           
                            # "Task_Achievement": {
                            #     "type": "string",
                            #     "description": "This assesses how well the student fulfills the requirements of the given task, including addressing all parts of the task, presenting a clear and focused response, and ensuring relevance to the essay.",
                            # },
                            "Task_Achievement_score":{
                                "type":"number",
                                "description":"Max score is 10, base on Task_Achievement of submited essay should score it"
                            },
                            "Task_Achievement_help":{
                                "type":"string",
                                "description":"Teacher recommend how to help student archive 10 score in future, base base on Task_Achievement of submited essay. Give some example to enhancement in submited essay"
                            },
                            # "Coherence_and_Cohesion": {
                            #     "type": "string",
                            #     "description": "This evaluates the clarity and organization of the writing, ensuring ideas are logically arranged and easy to follow. Cohesion involves using devices like linking words, pronouns, and conjunctions to connect ideas, sentences, and paragraphs",
                            # },
                            "Coherence_and_Cohesion_score":{
                                "type":"number",
                                "description":"Max score is 10, base on Coherence_and_Cohesion of submited essay should score it"
                            },
                            "Coherence_and_Cohesion_help":{
                                "type":"string",
                                "description":"Teacher recommend how to help student archive 10 score in future, base base on Coherence_and_Cohesion of submited essay. Give some example to enhancement in submited essay"
                            },
                            # "Lexical_Resource_Vocabulary": {
                            #     "type": "string",
                            #     "description": "This criterion focuses on the range and use of vocabulary in the writing, including the appropriateness, variety, and precision of word choice. Effective use of vocabulary enhances the quality of writing",
                            # },
                            "Lexical_Resource_Vocabulary_score":{
                                "type":"number",
                                "description":"Max score is 10, base on Lexical_Resource_Vocabulary of submited essay should score it"
                            },
                            "Lexical_Resource_Vocabulary_help":{
                                "type":"string",
                                "description":"Teacher recommend how to help student archive 10 score in future, base base on Lexical_Resource_Vocabulary of submited essay. Give some example to enhancement in submited essay"
                            },
                            # "Grammatical_Range_and_Accuracy": {
                            #     "type": "string",
                            #     "description": "This measures the writer's ability to use a range of grammatical structures accurately, assessing the ability to express ideas without errors that hinder communication. High grammatical control contributes to overall clarity and effectiveness of the writing",
                            # },
                            "Grammatical_Range_and_Accuracy_score":{
                                "type":"number",
                                "description":"Max score is 10, base on Grammatical_Range_and_Accuracy of submited essay should score it"
                            },
                            "Grammatical_Range_and_Accuracy_help":{
                                "type":"string",
                                "description":"Teacher recommend how to help student archive 10 score in future, base base on Grammatical_Range_and_Accuracy of submited essay. Give some example to enhancement in submited essay"
                            },
                            # ,
                            # "format": {
                            #     "type": "string",
                            #     "enum": ["celsius", "fahrenheit"],
                            #     "description": "The temperature unit to use. Infer this from the users location.",
                            # },
                            "Overall_grade":{
                                "type":"string",
                                "description":"Overall assessment. Teachers' comments must be friendly to help student understand why they got score for each criterias and be motivated to learn and strive to continue."
                            }
                        },
                        "required": [
                            "Task_Achievement",
                            "Coherence_and_Cohesion",
                            "Lexical_Resource_Vocabulary",
                            "Grammatical_Range_and_Accuracy",
                        ],
                    },
                },
            }
        ],
    )

    print(chat_completion)

    return chat_completion


qtext = """
Teacher assign essay for student as bellow:
#Question begin:
You received an email from your friend, Anna. She asked you some information about Hanoi. Read part of her email below.

I am looking for a place to spend my summer vacation and I am thinking about the city where you are living now. Can you give me some information about Hanoi (things like where to stay, how to get around, places to visit and things to do)? I want to see if I should come to Hanoi next month to enjoy my summer holiday. 

Write an email responding to Anna.
You should write at least 120 words. You do not need to include your name or addresses. Your response will be evaluated in terms of Task Fulfillment, Organization, Vocabulary and Grammar.
#Question end
Student Submited essay bellow:
#Essay begin:
Hi Anna,

I'm excited to hear that you're considering Hanoi for your summer vacation! It's a fantastic city with so much to offer.

For accommodation, I recommend staying in the Old Quarter or Hoan Kiem District. These areas are bustling with life, filled with charming streets and plenty of hotels, hostels, and guesthouses to choose from.

Getting around Hanoi is relatively easy. You can walk to many attractions in the Old Quarter and Hoan Kiem District. Alternatively, you can take a cyclo ride for a unique experience, or use taxis and ride-hailing services like Grab for longer distances.

As for places to visit, don't miss Hoan Kiem Lake, the Temple of Literature, and the Ho Chi Minh Mausoleum. And of course, exploring the street food scene is a must! Try dishes like pho, banh mi, and bun cha for an authentic taste of Vietnam.

I hope this helps you plan your trip. Let me know if you need any more information!

Best regards,
Quynh Anh
#Essay end

Teacher should grade score and explain how to help student improve submit easy for each criterials: Task_Achievement, Coherence_and_Cohesion, Lexical_Resource_Vocabulary, Grammatical_Range_and_Accuracy. Give some example to improve submited essay
"""
import json
res=grade_essay(qtext)

for choice in res.choices:
    for toolres in choice.message.tool_calls:
        print ( toolres.function.name)
        
        argsparsed= json.loads(toolres.function.arguments)
        
        print(argsparsed)
        


"""
Du vào mục 'CEFR" test: 

Highlights include:
•  Results available in <5 seconds (including Speaking and Writing)
•  Individualized feedback for each test taker
•  96% accurate when compared to a human evaluator
•  Fully mobile-responsive (tests can be taken on a cell phone, tablet or computer)

For the Writing test specifically, we provide feedback along the following 4 parameters:
•  Task Achievement: This assesses how well the writer fulfills the requirements of the given task, including addressing all parts of the task, presenting a clear and focused response, and ensuring relevance to the essay.
•  Coherence and Cohesion: This evaluates the clarity and organization of the writing, ensuring ideas are logically arranged and easy to follow. Cohesion involves using devices like linking words, pronouns, and conjunctions to connect ideas, sentences, and paragraphs.
•  Lexical Resource (Vocabulary): This criterion focuses on the range and use of vocabulary in the writing, including the appropriateness, variety, and precision of word choice. Effective use of vocabulary enhances the quality of writing.
•  Grammatical Range and Accuracy: This measures the writer's ability to use a range of grammatical structures accurately, assessing the ability to express ideas without errors that hinder communication. High grammatical control contributes to overall clarity and effectiveness of the writing.
"""
