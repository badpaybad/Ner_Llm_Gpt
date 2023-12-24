import random
import spacy
from spacy.training import Example
def train():
    LABEL_ho = 'Ho'
    LABEL_ten = 'Ten'
    LABEL_sdt = "Sdt"
    TRAIN_DATA = [
        ("Phạm Đỗ là bố bạn Nguyễn Phuc Lam", {'entities': [(0, 4, LABEL_ho),(5, 7, LABEL_ten)]}),
        ("Bo ban Phuc Lam ten la gi?", {'entities': []}),
        ("Phạm Đỗ la mot nhân viên lap trinh phan mem", {'entities': [(0, 4, LABEL_ho),(5, 7, LABEL_ten)]}),
        ("Phạm Đỗ thich di phuot", {'entities': [(0, 4, LABEL_ho),(5, 7, LABEL_ten)]}),
        ("Do la Phạm Đỗ, bo cua ban Phuc Lam", {'entities': [(6, 10, LABEL_ho),(10, 12, LABEL_ten)]}),
        ("So dien thoai 0778384839 hoac 84778384839", {'entities': [(14, 24, LABEL_sdt),(31, 42, LABEL_sdt )]} ),
        ("Phạm Đỗ?", {'entities': [(0, 4, LABEL_ho),(5, 7, LABEL_ten)]})
    ]
    nlp = spacy.load('xx_ent_wiki_sm')  # load existing spaCy model
    ner = nlp.get_pipe('ner')
    
    unique_labels = list(set(ent for ent in ner.labels))
    # Print all unique entity labels
    print("Unique Entity Labels in the Model:")
    print(unique_labels)    
    
    ner.add_label(LABEL_ho)
    ner.add_label(LABEL_ten)
    ner.add_label(LABEL_sdt)
    
    optimizer = nlp.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(100):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.25, sgd=optimizer, losses=losses)
            # print(text)
            # print(annotations)
            print(losses)

    nlp.to_disk("test_ho_ten")

    # # test the trained model ô nguyễn phan du
    # test_text = 'bo ban Lam ten la Phạm Du? The bo ban lam ten la gi?'
    # docf = nlp(test_text)
    # print("Entities in '%s'" % test_text)
    # for ent in docf.ents:
    #     print(ent.label_, " -- ", ent.text)
        
    # pip3 install spacy
    # python3 -m spacy download en_core_web_sm
    # python3 -m spacy download en_core_web_lg
    # python3 -m spacy download xx_ent_wiki_sm
    # python3 -m spacy download xx_sent_ud_sm


def test():    
    
    nlp = spacy.load('test_ho_ten')  # load existing spaCy model
    
    # Get all unique entity labels in the model
    unique_labels = list(set(ent for ent in nlp.get_pipe("ner").labels))

    # Print all unique entity labels
    print("Unique Entity Labels in the Model:")
    print(unique_labels)
    
    # test the trained model
    test_text = 'bo ban Lam ten la Phạm Đô? dien thoai: 0778384839'
    docf = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in docf.ents:
        
        print(ent.label_, " -- ", ent.text )
        
    
    pass
train()
test()