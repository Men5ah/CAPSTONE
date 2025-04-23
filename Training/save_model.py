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

    {
 "cell_type": "code",
 "metadata": {},
 "source": [
  "# Save best model (RNN) and preprocessing pipeline for deployment\n",
  "import pickle\n",
  "\n",
  "# Assuming rnn_model is the best model trained on clean data\n",
  "best_rnn_model = None\n",
  "for model_name, model in clean_results.items():\n",
  "    if model_name == 'RNN':\n",
  "        best_rnn_model = model\n",
  "        break\n",
  "\n",
  "if best_rnn_model is not None:\n",
  "    # Save the RNN model\n",
  "    best_rnn_model.save('rnn_bot_detection_model.h5')\n",
  "    print('RNN model saved as rnn_bot_detection_model.h5')\n",
  "else:\n",
  "    print('RNN model not found in clean_results')\n",
  "\n",
  "# Save the preprocessing pipeline\n",
  "with open('bot_detection_preprocessor.pkl', 'wb') as f:\n",
  "    pickle.dump(processed_data['clean']['preprocessor'], f)\n",
  "print('Preprocessing pipeline saved as bot_detection_preprocessor.pkl')"
 ]
}
{
 "cell_type": "code",
 "metadata": {},
 "source": [
  "# Save best model (RNN) and preprocessing pipeline for deployment\n",
  "import pickle\n",
  "\n",
  "# Assuming rnn_model is the best model trained on clean data\n",
  "best_rnn_model = None\n",
  "for model_name, model in clean_results.items():\n",
  "    if model_name == 'RNN':\n",
  "        best_rnn_model = model\n",
  "        break\n",
  "\n",
  "if best_rnn_model is not None:\n",
  "    # Save the RNN model\n",
  "    best_rnn_model.save('rnn_bot_detection_model.h5')\n",
  "    print('RNN model saved as rnn_bot_detection_model.h5')\n",
  "else:\n",
  "    print('RNN model not found in clean_results')\n",
  "\n",
  "# Save the preprocessing pipeline\n",
  "with open('bot_detection_preprocessor.pkl', 'wb') as f:\n",
  "    pickle.dump(processed_data['clean']['preprocessor'], f)\n",
  "print('Preprocessing pipeline saved as bot_detection_preprocessor.pkl')"
 ]
}

