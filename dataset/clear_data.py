import json

#Function to filter dataset and extract relevant fields
def filter_dataset(file_path):
    final_data = []
    with open(file_path, 'r') as file:
        question_counter = 1
        for line in file:
            data = json.loads(line)
            for k,v in data.items():
                if k == 'questions':
                    for question in v:
                        question_text = question.get('questionText', '').strip()
                        question_time = question.get('questionTime', '').strip()
                        question_type = question.get('questionType', '').strip()

                        question_answers = {
                            'question_id': question_counter,
                            'question_text': question_text,
                            'question_time': question_time,
                            'question_type': question_type,
                            'answers': []
                        }

                        for answer in question.get('answers', []):
                            answer_text = answer.get('answerText', '').strip()
                            answer_time = answer.get('answerTime', '').strip()
                            question_answers['answers'].append({
                                'answer_text': answer_text,
                                'answer_time': answer_time
                            })
                        final_data.append(question_answers)
                        question_counter += 1
    return final_data


#Save filtered data to a new JSONL file
def save_filtered_data(filtered_data, output_file_path):
    with open(output_file_path, 'w') as outfile:
        for entry in filtered_data:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    data = filter_dataset('dataset/QA_Video_Games.jsonl')
    save_filtered_data(data, 'dataset/filtered_QA_Video_Games.jsonl')
    print(json.dumps(data[0], indent=4))