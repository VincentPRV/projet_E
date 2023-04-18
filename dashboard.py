import joblib
from dash import Dash, html, dcc, Input, Output
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
from dash.dependencies import Input, Output, State

# Chargement du modèle de NLP et de classification
with open('estimator_model.pkl', 'rb') as f:
    loaded_model = joblib.load(f)

def cleanContent(review_text):
    stop_words =stopwords.words('french')
    new_stopwords_to_add= ['allociné', '892', '000', 'cookies']
    stop_words.extend(new_stopwords_to_add)
    review_text = str(review_text).lower().strip()
    review_text = word_tokenize(review_text)
    review_text = [word for word in review_text if word not in stop_words]
    lemma=WordNetLemmatizer()
    review_text = [lemma.lemmatize(word=w, pos='v') for w in review_text]
    review_text = [w for w in review_text if len(w) > 2]
    review_text = ' '.join(review_text)
    return [review_text]


# Création de l'application Dash
app = Dash(__name__)

# Mise en page de l'application
app.layout = html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                      children=[
    html.H1("Bienvenue sur l'application \"D'où provient mon article de journal?\""),
    html.H3("Veuillez entrer le titre d'un article, et je vous dirais quel journal pourrait l'avoir écrit."),
    html.Br(),
    html.Br(),
    dcc.Textarea(
        id='article-titre',
        value='',
        placeholder='Entrez le titre de l\'article ici',
        style={
            'width': '70%',
            'height': '120px',
            'font-size': '20px',
            'border-radius': '5px',
            'border': '2px solid #cccccc',
            'padding': '10px',
            'resize': 'none',
            'margin': 'auto'
        }
    ),
    html.Br(),
    html.Br(),
    html.Button('Valider', id='valider'),
    html.Br(),
    html.Br(),
    html.Div(id='output', style={'text-align': 'center'})
])

# Callback pour afficher le résultat de la prédiction
@app.callback(
    Output(component_id='output', component_property='children'),
    Input(component_id='valider', component_property='n_clicks'),
    State(component_id='article-titre', component_property='value')
)

def update_output(n_clicks, value):
    if n_clicks is not None and value:
        titre = cleanContent(value)
       
        # Prédiction du modèle
        y_pred_proba = loaded_model.predict_proba(titre)
        y_pred = loaded_model.predict(titre)
        print('y_pred_proba :', y_pred_proba)
        print('y_pred :', y_pred)
        
        # Score correspondant à la classe prédite
        score_max = y_pred_proba.max()
        index_max = y_pred_proba.argmax()
        score_second = sorted(y_pred_proba[0])[-2]
        index_second = list(y_pred_proba[0]).index(score_second)
        source_second = loaded_model.classes_[index_second]

        return html.Div(style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'},
                        children=[html.H3(f"L'article que vous avez entré a probablement été écrit par {loaded_model.classes_[index_max]} avec un score de {score_max:.2f}.", style={'font-weight': 'bold', 'font-size': '25px'}), html.Br(), html.H4("Autres possibilités :", style={'font-style': 'italic', 'font-size': '20px'}), html.P([f"{loaded_model.classes_[i]} avec un score de {proba:.2f}." for i, proba in enumerate(y_pred_proba[0]) if proba != score_max])])
    else:
        return ""


if __name__ == '__main__':
    app.run_server(debug=True)
