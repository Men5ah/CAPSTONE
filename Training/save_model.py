import pickle

def save_rnn_model_and_preprocessor(clean_results, processed_data):
    best_rnn_model = None
    for model_name, model in clean_results.items():
        if model_name == 'RNN':
            best_rnn_model = model
            break

    if best_rnn_model is not None:
        # Save the RNN model
        best_rnn_model.save('rnn_bot_detection_model.h5')
        print('RNN model saved as rnn_bot_detection_model.h5')
    else:
        print('RNN model not found in clean_results')

    # Save the preprocessing pipeline
    with open('bot_detection_preprocessor.pkl', 'wb') as f:
        pickle.dump(processed_data['clean']['preprocessor'], f)
    print('Preprocessing pipeline saved as bot_detection_preprocessor.pkl')
