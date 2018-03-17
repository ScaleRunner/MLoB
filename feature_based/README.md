

# Structure

To add a new classifier, simply copy the random forest or MLP and switch the adopt the create_model and train_model to use the new classifier.

# Feature data

Since extracting the full feature set takes about ten minutes, I added utilities to save them to a csv.

These utils are contained in the utils file.

I also moved the data handling functions to the utils file, so each classifier file can simply import them (when adding new functions to utils, you have to import them in the \_\_init__.py file to use them).


# Results
The random forest can correctly predict __all__ classes in about 90% of the cases, so this is not a really good result. But there might be some room for improvement by changing the parameters.