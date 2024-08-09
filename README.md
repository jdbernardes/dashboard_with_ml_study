# Dashboard With ML Study

## Introduction

The following project aims to apply the usage of some technologies to deploy a dashboard app.
The dashboard will be developed with Streamlit whereas the data will be used consumed from Kaggle and persisted in a relational database.
Below is a list of some of the technologies that will be applied:

    1. Docker: will be used to allow the creation of containers in 2 different phases

        1.1. First will be used to deploy a postgres SQL and pgadmin service

        1.2. Secondly it will be used to deploy the full app all in the same container

    2. Streamlit: Streamlit is a library that allows you to build web applications in python. This lib allows you to not only create dashboards but to create general web applications.
    
    3. Scikit Learn: for the ML algoritm we are going to make use of the scikit lear library, one of the most famous ML libs we have.

## APP