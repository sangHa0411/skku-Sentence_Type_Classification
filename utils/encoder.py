class Encoder :

    def __init__(self, tokenizer, max_input_length, label_dict=None, train_flag=True) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.label_dict = label_dict
        self.train_flag = train_flag

    def __call__(self, examples) :

        inputs = examples['문장']
        model_inputs = self.tokenizer(inputs, 
            max_length=self.max_input_length, 
            return_token_type_ids=False,
            truncation=True,
        )

        if self.train_flag :
            model_inputs['labels1'] = [self.label_dict['유형'][l] for l in examples['유형']]
            model_inputs['labels2'] = [self.label_dict['극성'][l] for l in examples['극성']]
            model_inputs['labels3'] = [self.label_dict['시제'][l] for l in examples['시제']]
            model_inputs['labels4'] = [self.label_dict['확실성'][l] for l in examples['확실성']]

        return model_inputs