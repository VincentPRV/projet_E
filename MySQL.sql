--  Création de la database
CREATE DATABASE IF NOT EXISTS projetjournaux;

USE projetjournaux;


--  Création de la table "journal" 
CREATE TABLE IF NOT EXISTS journal(
    nom   VARCHAR(150) NOT NULL,
    parution   VARCHAR(150) NOT NULL,
    UNIQUE(nom)
);


--  Création de la table "articles" 
CREATE TABLE IF NOT EXISTS articles(
    titre   VARCHAR(500),
    content   TEXT NOT NULL,
    journal    VARCHAR(500) NOT NULL, 
    foreign key(journal) references journal(nom),
    UNIQUE(titre) 
);

