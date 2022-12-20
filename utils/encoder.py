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
            model_inputs['labels'] = [self.label_dict[l] for l in examples['label']]

        return model_inputs