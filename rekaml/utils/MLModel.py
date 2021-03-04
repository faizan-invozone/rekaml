from pathlib import Path
import shutil
import os

def create_model(directory, is_new=True, tuning_file=None):
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        print('starting frist step.')
        import gensim.downloader as api
        glove = api.load("glove-twitter-100")
        print('frist step ended.')
        ####################################
        print('starting second step.')
        import random
        import numpy as np

        import nltk
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        STOPWORDS = stopwords.words("english")
        # word tokenization
        from nltk.tokenize import word_tokenize
        nltk.download('punkt')

        import keras
        from keras import backend as K
        from keras.models import load_model
        print('second step ended.')

        #######################################

        print('starting third step.')
        # Dictionary Utils
        TOP_NUM = 50
        '''
        If this variable changews, must re-train the model
        '''
        MAX_WORD_SPACE = 15000

        class TwoWayDict(dict):
            def __len__(self):
                return dict.__len__(self) // 2
            def __setitem__(self, key, value):
                dict.__setitem__(self, key, value)
                dict.__setitem__(self, value, key)

        '''
        Load default dictionary from file
        '''        
        def load_default_dict(default_dict_file='{}/utils/dictionary.txt'.format(BASE_DIR)):
            dic = TwoWayDict()
            file = open(default_dict_file, 'r').read().splitlines()
            for line, idx in zip(file, range(len(file))):
                dic[line.rstrip()] = idx
            return dic

        '''
        Add a new word to the dictionary, note it won't be saved
        to the default dictionary unless update_default_dict() is
        called
        '''
        def add_new_word(dic, word, freq_count = []):
            l = len(dic)
            if l < MAX_WORD_SPACE:
                dic[l] = word
                return dic
            else:
                if not bool(freq_count): # loading stage
                    rint = random.randint(0, MAX_WORD_SPACE - 1)
                    oldword = dic[rint]
                    dic.update({rint: word})
                    newword = dic[rint]
                    #print("replaced", oldword, "with", newword)
                    return dic
                else: #actually exploding
                    keys = sorted(range(len(freq_count)), key=lambda k: freq_count[k])
                    cutting = len(keys)
                    if len(keys) > 10:
                        cutting = 10
                    keys = keys[:cutting]
                    rint = keys[random.randint(0, cutting - 1)]
                    oldword = dic[rint]
                    dic.update({rint: word})
                    dic.update({word: rint})
                    newword = dic[rint]
                    #print("replaced", oldword, "with", newword)
                    #print("cutting", cutting, "rint", rint)
                    return dic
                
        # def add_new_word(dic, word, freq_count=[]):
        #     # if the word already exists in dic, return
        #     if word in dic:
        #         return dic
        #     len_dic = len(dic)
        #     # if we have space in dictionary and its not a stopword, add
        #     if len_dic < MAX_WORD_SPACE and word not in STOPWORDS:
        #         dic[len_dic] = word
        #         return dic
        #     # loading stage
        #     elif not bool(freq_count):
        #         print("LOADING STAGE")
        #         rint = random.randint(0, word_space - 1)
        #         oldword = dic[rint]
        #         dic.update({rint: word})
        #         newword = dic[rint]
        #         return dic        
        #     # if we don't have space in dictionary, replace with some low
        #     # frequency word
        #     elif len_dic == MAX_WORD_SPACE and word not in STOPWORDS:
        #         keys = sorted(range(len(freq_count)), key=lambda k: freq_count[k])
        #         cutting = len(keys)
        #         if len(keys) > 10:
        #             cutting = 10
        #         keys = keys[:cutting]
        #         rint = keys[random.randint(0, cutting - 1)]
        #         dic.update({rint: word})
        #         dic.update({word: rint})
        #         return dic

        '''
        Function to overwrite the default dictionary file
        '''
        def update_default_dict(dic, default_dict_file='{}/utils/dictionary.txt'.format(BASE_DIR)):
            new_default_dict_file = open(default_dict_file, 'w')
            for idx in range(dic.__len__()):
                new_default_dict_file.write(dic[idx] + '\n')


        print('third step ended.')

        ##############################################
        print('starting fourth step.')
        # Structures to Maintain
        WORD_DICTIONARY = load_default_dict()
        TAG_DICTIONARY = {}
        TAG_VECTORS = np.array([])
        USED_COUNT = [0] * MAX_WORD_SPACE

        print('fourth step ended.')
        ##############################################

        print('starting fifth step.')
        # Fine-tuning Utils
        def predict(tag_vectors, current_tags, d, td, show_details = False):
            output_vector = np.zeros((1, MAX_WORD_SPACE))
            for tag in current_tags:
                output_vector = np.add(output_vector, tag_vectors[td[tag]])
            output_vector = output_vector.T.reshape((MAX_WORD_SPACE,))
            s = np.sum(output_vector)
            output_vector = output_vector / s
            return output_vector

        def update(tag_vectors, current_tags, word, d, td, learning_rate):
            for tag in current_tags:
                tag_vectors[td[tag], d[word]] += learning_rate #corrected on ms6-2
                s = np.sum(tag_vectors[td[tag]])
                tag_vectors[td[tag]] = tag_vectors[td[tag]] / s
            return tag_vectors

        def one_step_training(tag_vectors, current_tags, word, dic, td, learning_rate, show_details = False):
            probs = predict(tag_vectors, current_tags, dic, td, show_details)
            if word not in dic:
                dic = add_new_word(dic, word)
            tv = update(tag_vectors, current_tags, word, dic, td, learning_rate)
            one_hot_vec = np.zeros(MAX_WORD_SPACE)
            one_hot_vec[dic[word]] = 1
            ev = np.dot(probs, one_hot_vec)/(np.linalg.norm(probs)*np.linalg.norm(one_hot_vec)) #cosine similarity
            return tv, ev

        def print_top_n(tag_vectors, current_tags, d, td, n):
            output_vector = np.zeros((1, MAX_WORD_SPACE))
            for tag in current_tags:
                output_vector = np.add(output_vector, tag_vectors[td[tag]])
            output_vector = output_vector.T.reshape((MAX_WORD_SPACE,))
            s = np.sum(output_vector)
            output_vector = output_vector / s
            top_n_idx = np.argsort(output_vector)[-n:]
            top_n_values = [output_vector[i] for i in top_n_idx]
            for i in reversed(range(n)):
                print(n - i, ":", d[top_n_idx[i]], top_n_values[i])
                
        def gen_tag_vector(word, dic, used_count):  
            # returns a numpy array in shape of (1, MAX_WORD_SPACE)
            tv = np.zeros((1, MAX_WORD_SPACE))
            try:
                wv = glove.most_similar(word, topn = TOP_NUM) 
            except:
                #updated on ms6-2
                random_tag_vector = np.append(np.random.rand(1, len(dic)), np.zeros((1, MAX_WORD_SPACE - len(dic)))).reshape(1, MAX_WORD_SPACE)
                s = np.sum(random_tag_vector)
                random_tag_vector = random_tag_vector / s
                return random_tag_vector
            
            for w, val in wv:
                if w not in dic:
                    #updated on ms6-2
                    dic = add_new_word(dic, w, used_count)
                tv[0, dic[w]] = val
            s = np.sum(tv)
            tv = tv / s
            
            #added in ms6, updated in ms6-2
            if word not in dic:
                dic = add_new_word(dic, word, used_count)
            tv[0, dic[word]] = 0.02
            s = np.sum(tv)
            tv = tv / s
            return tv

        '''
        Fine tune the model, save an updated model file and return the
        structures needed for inference
        '''
        def fine_tune_model(training_data, spectate_list, pred_rate=0.8, top_n=10, lstm_rt_update=False, p_learning_rate=0.08, p_epoch=1, model_name='{}/utils/lstm256-switchboard-300-epoch-lr003.h5'.format(BASE_DIR)):
            dictionary = WORD_DICTIONARY
            tag_vectors = np.array([])
            tag_dictionary = {}
            used_count = USED_COUNT
            
            evaluates = []
            learning_rate = 0.02
            model = load_model(model_name)
            K.set_value(model.optimizer.learning_rate, p_learning_rate)
            
            with open(training_data) as f:    
                test_num = 0
                for line in f:
                    if line[0] == '{':
                        test_num += 1
                        evals = []
                        cur_tags = []
                        pred_window = np.array((-1, -1, -1)).reshape(1, 3)
                        if test_num in spectate_list:
                            spectated = True
                        else:
                            spectated = False
                    elif line[0] == '#':
                        line = line[1:]
                        for word in line.split():
                            word = word.lower()
                            if word not in tag_dictionary:
                                tag_dictionary[word] = len(tag_dictionary)
                                #updated on ms6-2
                                if tag_vectors.shape[0] == 0:
                                    tag_vectors = gen_tag_vector(word, dictionary, used_count)
                                else:
                                    tag_vectors = np.vstack((tag_vectors, gen_tag_vector(word, dictionary, used_count)))     
                            cur_tags.append(word)
                        if spectated:
                            print(cur_tags)
                            print("Before:")
                            print_top_n(tag_vectors, cur_tags, dictionary, tag_dictionary, n=10)
                    elif line[0] == '}':
                        if len(evals) != 0:
                            evaluate = sum(evals) / len(evals)
                            evaluates.append(evaluate)
                        if spectated:
                            print("After:")
                            print_top_n(tag_vectors, cur_tags, dictionary, tag_dictionary, n=10)
                    else:
                        text_tokens = word_tokenize(line)
                        text_tokens = [word.lower() for word in text_tokens]
                        tokens_without_sw = [word for word in text_tokens if not word in STOPWORDS]
                        words = [word for word in tokens_without_sw if word.isalnum()]
                        words_w_sw = [word for word in text_tokens if word.isalnum()]
                        for word in words_w_sw:
                            #updated on ms6-2
                            if word not in dictionary:
                                dictionary = add_new_word(dictionary, word, used_count)
                            used_count[dictionary[word]] += 1
                            if used_count[dictionary[word]] == 10000:
                                for i in range(MAX_WORD_SPACE):
                                    used_count[i] = used_count[i] / 10
                                
                            #make prediction
                            reco_vec = predict(tag_vectors, cur_tags, dictionary, tag_dictionary)
                            if pred_window[0][0] != -1:
                                pred_vec = model.predict(pred_window).reshape(MAX_WORD_SPACE,)
                                output_vec = reco_vec * (1 - pred_rate) + pred_vec * pred_rate
                            else:
                                output_vec = reco_vec
                            output = output_vec.argsort()[::-1][:top_n]
                            
                            #spectate the output
                            if spectated:
                                output_words = []
                                for i in output:
                                    if i < len(d):
                                        output_words.append(dictionary[i])
                                print(output_words)
                            
                            #evaluate
                            if dictionary[word] in output:
                                evals.append((top_n - np.where(output == dictionary[word])[0]) / top_n * 20 + 80) #100 ... 80 if hit
                                if spectated:
                                    print("HIT", word, np.where(output == dictionary[word])[0][0])
                            else:
                                evals.append(0) # 0 if miss
                                
                            #update model
                            if pred_window[0][0] != -1 and lstm_rt_update:
                                print("got here")
                                y = np.array(dictionary[word]).reshape(1, 1)
                                y = to_categorical(y, num_classes=MAX_WORD_SPACE)
                                model.fit(pred_window, y, epochs = p_epoch, verbose = 0)
                            if word not in STOPWORDS:
                                tag_vectors, ev = one_step_training(tag_vectors, cur_tags, word, dictionary, tag_dictionary, learning_rate, show_details=spectated)
                            pred_window[0][0] = pred_window[0][1]
                            pred_window[0][1] = pred_window[0][2]
                            pred_window[0][2] = dictionary[word]
            
            dir = '{}/models/{}/1'.format(BASE_DIR, directory)
            if os.path.exists(dir):
                new_dir = '{}/models/{}/3'.format(BASE_DIR, directory)
                shutil.move(dir, new_dir)
            else:
                os.makedirs(dir)
            # dir = '{}/models/{}/'.format(BASE_DIR, directory)
            # if not os.path.exists(dir):
            #     os.mkdir(dir)
            try:
                model.save('{}/models/{}/1/newmodel.h5'.format(BASE_DIR, directory))
                shutil.move(tuning_file, dir)
                dir = '{}/models/{}/2'.format(BASE_DIR, directory)
                if os.path.exists(dir):
                    shutil.rmtree(dir, ignore_errors=True)
                    # os.makedirs(dir)
                    new_dir = '{}/models/{}/3'.format(BASE_DIR, directory)
                    shutil.move(new_dir, dir)
                else:
                    os.makedirs(dir)
                    new_dir = '{}/models/{}/3'.format(BASE_DIR, directory)
                    shutil.move(new_dir, dir)
            except Exception as e:
                new_dir = '{}/models/{}/3/'.format(BASE_DIR, directory)
                shutil.move(new_dir, dir)

            print("Averaged Score:", sum(evaluates) / len(evaluates))
            return tag_vectors, dictionary, tag_dictionary, used_count

        
        print('fith step ended.')
        ########################################

        # remove directory 2 if exists
        # rename directory 1 to 2 if exists
        # create new directory 1 to store model against the user and relevant files
        # move tuning file to directory 1


        print('starting sixth step.')
            # Call to update the model
        file_for_tuning = "{}/utils/ms5-test-a.txt".format(BASE_DIR)
        if not is_new:
            file_for_tuning = tuning_file
        tag_vectors, dictionary, tag_dictionary, used_count = fine_tune_model(file_for_tuning, [], 0.01)
        tag_vectors_dir = '{}/models/{}/1/tag_vectors.txt'.format(BASE_DIR, directory)
        if os.path.isfile(tag_vectors_dir):
            os.remove(tag_vectors_dir)
        with open(tag_vectors_dir, "w+") as output:
            for row in tag_vectors:
                np.savetxt(output, row)    
            # output.write(str(tag_vectors))
        dictionary_dir = '{}/models/{}/1/dictionary.txt'.format(BASE_DIR, directory)
        if os.path.isfile(dictionary_dir):
            os.remove(dictionary_dir)
        with open(dictionary_dir, "w+") as output:
            output.write(str(dictionary))
        tag_dictionary_dir = '{}/models/{}/1/tag_dictionary.txt'.format(BASE_DIR, directory)
        if os.path.isfile(tag_dictionary_dir):
            os.remove(tag_dictionary_dir)
        with open(tag_dictionary_dir, "w+") as output:
            output.write(str(tag_dictionary))
        used_count_dir = '{}/models/{}/1/used_count.txt'.format(BASE_DIR, directory)
        if os.path.isfile(used_count_dir):
            os.remove(used_count_dir)
        with open(used_count_dir, "w+") as output:
            output.write(str(used_count))
        print('sixth step ended.')

        ###################################################

        # Call to run inference
        print('printing suggestions')
        current_tags = ['roommate_jen', 'netflix']
        print_top_n(tag_vectors, current_tags, dictionary, tag_dictionary, n=5)
    except Exception as e:
        print(str(e))