from asari.api import Sonar
import os
sonar = Sonar()

data = []
for filename in os.listdir('text'):
    doc_box = []
    f = open('text/' + filename, 'r')
    for row in f:
        row = row.rstrip()
        doc_box.append(row)
    result_doc = ''.join(doc_box)
    res = sonar.ping(text=result_doc)
    classes = res['classes']
    positive_confidence = next((item['confidence'] for item in classes if item['class_name'] == 'positive'), 0)
    negative_confidence = next((item['confidence'] for item in classes if item['class_name'] == 'negative'), 0)

    scaled_confidence = (positive_confidence - negative_confidence)
    data.append(scaled_confidence)
print(data)





