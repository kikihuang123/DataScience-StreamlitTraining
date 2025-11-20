{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyOn00ishHFnTks094ghmDsU",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kikihuang123/DataScience-StreamlitTraining/blob/main/model_training.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XsY4e6ljLVmV",
        "outputId": "286a0b35-c3f5-4a85-9441-6c7e74259c57"
      },
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(25000, 2)\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt.zip.\n",
            "[nltk_data] Downloading package punkt_tab to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt_tab.zip.\n",
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m47s\u001b[0m 4ms/step - accuracy: 0.0223 - loss: 7.9311\n",
            "Epoch 2/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.0552 - loss: 7.0153\n",
            "Epoch 3/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 4ms/step - accuracy: 0.0753 - loss: 6.5155\n",
            "Epoch 4/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.0903 - loss: 6.1529\n",
            "Epoch 5/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 4ms/step - accuracy: 0.1041 - loss: 5.8549\n",
            "Epoch 6/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1164 - loss: 5.6111\n",
            "Epoch 7/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 4ms/step - accuracy: 0.1281 - loss: 5.4129\n",
            "Epoch 8/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 4ms/step - accuracy: 0.1387 - loss: 5.2519\n",
            "Epoch 9/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1495 - loss: 5.1194\n",
            "Epoch 10/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1594 - loss: 5.0094\n",
            "Epoch 11/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1678 - loss: 4.9260\n",
            "Epoch 12/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1749 - loss: 4.8459\n",
            "Epoch 13/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1823 - loss: 4.7818\n",
            "Epoch 14/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1885 - loss: 4.7274\n",
            "Epoch 15/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1936 - loss: 4.6816\n",
            "Epoch 16/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.1987 - loss: 4.6399\n",
            "Epoch 17/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.2040 - loss: 4.5995\n",
            "Epoch 18/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.2077 - loss: 4.5680\n",
            "Epoch 19/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.2105 - loss: 4.5382\n",
            "Epoch 20/20\n",
            "\u001b[1m12163/12163\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 4ms/step - accuracy: 0.2144 - loss: 4.5128\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "\n",
        "df = pd.read_csv(\"/content/MovieReview.csv\")\n",
        "\n",
        "print(df.shape)\n",
        "\n",
        "df = df.drop('sentiment', axis=1)\n",
        "\n",
        "import re\n",
        "import unicodedata\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize\n",
        "\n",
        "nltk.download('punkt')\n",
        "nltk.download('punkt_tab')\n",
        "nltk.download('stopwords')\n",
        "stop_words = stopwords.words('english')\n",
        "\n",
        "# Converts the unicode file to ascii\n",
        "def unicode_to_ascii(s):\n",
        "    return ''.join(c for c in unicodedata.normalize('NFD', s)\n",
        "        if unicodedata.category(c) != 'Mn')\n",
        "\n",
        "def preprocess_sentence(w):\n",
        "    w = unicode_to_ascii(w.lower().strip())\n",
        "    # creating a space between a word and the punctuation following it\n",
        "    # eg: \"he is a boy.\" => \"he is a boy .\"\n",
        "    w = re.sub(r\"([?.!,¿])\", r\" \\1 \", w)\n",
        "    w = re.sub(r'[\" \"]+', \" \", w)\n",
        "    # replacing everything with space except (a-z, A-Z, \".\", \"?\", \"!\", \",\")\n",
        "    w = re.sub(r\"[^a-zA-Z?.!]+\", \" \", w)\n",
        "    w = re.sub(r'\\b\\w{0,2}\\b', '', w)\n",
        "\n",
        "    # remove stopword\n",
        "    mots = word_tokenize(w.strip())\n",
        "    mots = [mot for mot in mots if mot not in stop_words]\n",
        "    return ' '.join(mots).strip()\n",
        "\n",
        "df.review = df.review.apply(lambda x :preprocess_sentence(x))\n",
        "df.head()\n",
        "\n",
        "import tensorflow as tf\n",
        "tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)\n",
        "tokenizer.fit_on_texts(df.review)\n",
        "\n",
        "word2idx = tokenizer.word_index\n",
        "idx2word = tokenizer.index_word\n",
        "vocab_size = tokenizer.num_words\n",
        "\n",
        "import numpy as np\n",
        "\n",
        "\n",
        "def sentenceToData(tokens, WINDOW_SIZE):\n",
        "    window = np.concatenate((np.arange(-WINDOW_SIZE,0),np.arange(1,WINDOW_SIZE+1)))\n",
        "    X,Y=([],[])\n",
        "    for word_index, word in enumerate(tokens) :\n",
        "        if ((word_index - WINDOW_SIZE >= 0) and (word_index + WINDOW_SIZE <= len(tokens) - 1)) :\n",
        "            X.append(word)\n",
        "            Y.append([tokens[word_index-i] for i in window])\n",
        "    return X, Y\n",
        "\n",
        "\n",
        "WINDOW_SIZE = 5\n",
        "\n",
        "X, Y = ([], [])\n",
        "for review in df.review:\n",
        "    for sentence in review.split(\".\"):\n",
        "        word_list = tokenizer.texts_to_sequences([sentence])[0]\n",
        "        if len(word_list) >= WINDOW_SIZE:\n",
        "            Y1, X1 = sentenceToData(word_list, WINDOW_SIZE//2)\n",
        "            X.extend(X1)\n",
        "            Y.extend(Y1)\n",
        "\n",
        "X = np.array(X).astype(int)\n",
        "y = np.array(Y).astype(int).reshape([-1,1])\n",
        "\n",
        "from tensorflow.keras import Sequential\n",
        "from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D\n",
        "\n",
        "embedding_dim = 300\n",
        "model = Sequential()\n",
        "model.add(Embedding(vocab_size, embedding_dim))\n",
        "model.add(GlobalAveragePooling1D())\n",
        "model.add(Dense(vocab_size, activation='softmax'))\n",
        "\n",
        "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
        "model.fit(X, y, batch_size = 128, epochs=20)\n",
        "\n",
        "model.save(\"word2vec.h5\")\n",
        "\n",
        "import pickle\n",
        "vectors = model.layers[0].get_weights()[0]   # shape: (vocab_size, embedding_dim)\n",
        "np.save(\"vectors.npy\", vectors)\n",
        "\n",
        "with open(\"word2idx.pkl\", \"wb\") as f:\n",
        "    pickle.dump(word2idx, f)\n",
        "\n",
        "with open(\"idx2word.pkl\", \"wb\") as f:\n",
        "    pickle.dump(idx2word, f)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls -lh /content"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iEzt8nZiZbWq",
        "outputId": "928239be-b2bd-4442-a9c4-8e53cfcd12b9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "total 115M\n",
            "-rw-r--r-- 1 root root 977K Nov 20 22:17 idx2word.pkl\n",
            "-rw-r--r-- 1 root root  32M Nov 20 21:40 MovieReview.csv\n",
            "-rw-r--r-- 1 root root  13M Nov 20 21:39 MovieReview.csv.zip\n",
            "drwxr-xr-x 1 root root 4.0K Nov 17 14:29 sample_data\n",
            "-rw-r--r-- 1 root root 977K Nov 20 22:17 word2idx.pkl\n",
            "-rw-r--r-- 1 root root  69M Nov 20 22:17 word2vec.h5\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "id": "4sI6i6LqZmEN"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}