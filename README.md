kaggle-bulldozer
================

Model for the Blue Book for Bulldozers Challenge on Kaggle.com

Uses:
- python 2.7.1
- scikit-learn 0.13.1
- pymongo 2.1.1 (MongoDB is used for data storage)
- pandas 0.10.0

Code ran on:
- core i7, 4GB RAM
- needs around a day to train+test

Training and testing
- first data needs to be put in MongoDB: run script init_db.py
- training and testing are done in one step: run script test_model.py
    - why? because it was faster to develop like this and time was essential

More info: http://webmining.olariu.org/trees-ridges-and-bulldozers-made-in-1000-ad/
