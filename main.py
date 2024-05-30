#Copyright 2024 Pal Roberto-Giulio

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import re
import seaborn as sns

def task1():
    file = pd.read_csv("../Date/train.csv")
    cols = file.shape[1]
    print("Number of columns:", cols)
    print("")
    print("Data type across each column:")
    print(file.dtypes)
    print("")
    print(file.isnull().sum())
    print("")
    lines = file.shape[0]
    print("Number of lines of the file:", lines)
    print("")
    duplicates = file.duplicated().sum()
    if duplicates:
        print("The duplicated lines are:")
        print(duplicates)
        print("")
    else:
        print("There are no duplicated lines!")
        print("")


def task2():
    file = pd.read_csv("../Date/train.csv")
    lines = file.shape[0]
    survived = file['Survived'].value_counts() * 100 / lines # or normalize = true in the function
    print("Number of people that have survived: ", survived[0])
    print("Number of people that have died:", survived[1])
    print("")
    class_stats = file['Pclass'].value_counts() * 100 / lines
    print("Percentage of passangers in each class:")
    print(class_stats)
    print("")
    gender_stats = file['Sex'].value_counts() * 100 / lines
    print("Percentage of men and women:")
    print(gender_stats)
    print("")

    #create 3 subplots for each histogram
    graph, sub = plt.subplots(3, 1, figsize = (9, 12))

    # graph for survival ratio
    sub[0].bar(survived.index, survived.values, color = ['red', 'green'])
    sub[0].set_title("Alive vs Dead percentages")
    sub[0].set_xticks([1, 0])
    sub[0].set_xticklabels(["Alive", "Dead"])
    sub[0].set_ylabel("Percentage")

    # graph for repartition in classes
    sub[1].bar(class_stats.index, class_stats.values, color = ['yellow', 'orange', 'purple'])
    sub[1].set_title("Repartition of each class")
    sub[1].set_xticks([3, 2, 1])
    sub[1].set_xticklabels(["Class 3", "Class 2", "Class 1"])
    sub[1].set_ylabel("Percentage")

    # graph for gender distribution
    sub[2].bar(gender_stats.index, gender_stats.values, color = ['blue', 'red'])
    sub[2].set_title("Male vs Female percentages")
    sub[2].set_xticklabels(["Male", "Female"])
    sub[2].set_ylabel("Percentage")

    plt.tight_layout()
    plt.show()


def task3():
    file = pd.read_csv("../Date/train.csv")
    numbered_cols = file.select_dtypes(include="number").columns
    for column in numbered_cols:
        plt.figure(figsize=(6, 9))
        plt.hist(file[column].dropna(), ages=30, alpha=0.7)
        plt.title(f'Histrogram for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

def task4():
    file = pd.read_csv("../Date/train.csv")
    no_col = file.isnull().sum()
    no_col = no_col[no_col > 0] # keep only columns that have missing values
    lines = file.shape[0]
    print("Number and proportion of missing values:")
    for column, i in no_col.items():
        proportion = i / lines * 100
        print(f"{column}: {i} missing values ({proportion:.2f}%)")
    print("")
    print("Percentage of missing values based on survival rate:")
    for column in no_col.index:
        survived_missing = file[file[column].isnull()].groupby('Survived').size()
        for survived, i in survived_missing.items():
            if file[file['Survived'] == survived].shape[0]:
                proportion = i / file[file['Survived'] == survived].shape[0] * 100
                print(f"{column} (Survived={survived}): {i} missing values ({proportion:.2f}%)")

def task5():
    file = pd.read_csv("../Date/train.csv")
    ages = [0, 20, 40, 60, float('inf')]
    age_cat = ['0-20', '21-40', '41-60', '61+']
    file['AgeCategory'] = pd.cut(file['Age'], bins=ages, labels=age_cat, right=False)
    age_category = file['AgeCategory'].value_counts().sort_index()
    print("Number of passengers for each age category:")
    print(age_category)

    # Save the modified dataset to a new file
    file.to_csv("../Date/task5.csv", index=False)

    plt.figure(figsize=(10, 6))
    age_category.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Age distribution among passengers')
    plt.xlabel('Age category')
    plt.ylabel('Number of passengers')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()

def task6():
    file = pd.read_csv("../Date/train.csv")
    ages = [0, 20, 40, 60, float('inf')]
    age_cat = ['0-20', '21-40', '41-60', '61+']
    file['AgeCategory'] = pd.cut(file['Age'], bins=ages, labels=age_cat, right=False)

    men = file[file['Sex'] == 'male']
    survived_men = men[men['Survived'] == 1]
    survived_counts = survived_men['AgeCategory'].value_counts().sort_index()
    total_counts = men['AgeCategory'].value_counts().sort_index()
    survival_rate = (survived_counts / total_counts * 100).fillna(0)

    print("Percentage of male survival for each age group:")
    print(survival_rate)
    plt.figure(figsize=(10, 6))
    survival_rate.plot(kind='bar', color='red', alpha=0.7)
    plt.title('Percentage of male survival for each age category')
    plt.xlabel('Age categoty')
    plt.ylabel('Survival rate (%)')
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()

def task7():
    file = pd.read_csv("../Date/train.csv")
    file['IsKid'] = file['Age'] < 18 #filter to keep only people younger than 18
    total_passengers = len(file)
    total_children = file['IsKid'].sum()
    children_percentage = (total_children / total_passengers) * 100
    print(f"Children percentage on board: {children_percentage:.2f}%")
    survival_rate_children = file[file['IsKid'] & file['Survived'] == 1].shape[0] / total_children * 100
    survival_rate_adults = file[~file['IsKid'] & file['Survived'] == 1].shape[0] / (total_passengers - total_children) * 100

    print(f"Survival rate for children: {survival_rate_children:.2f}%")
    print(f"Survival rate for adults: {survival_rate_adults:.2f}%")
    categories = ['Children', 'Adults']
    survival_rates = [survival_rate_children, survival_rate_adults] #let's compare survival rates

    plt.figure(figsize=(10, 6))
    plt.bar(categories, survival_rates, color=['skyblue', 'lightcoral'], edgecolor='k', alpha=0.7)
    plt.title('Survival rate')
    plt.xlabel('Category')
    plt.ylabel('Survival rate (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()

def task8():
    file = pd.read_csv("../Date/train.csv")
    def missing_num(file, column, groupby_column):
        mean_values = file.groupby(groupby_column)[column].mean()
        file[column] = file.apply(lambda row: mean_values[row[groupby_column]] if pd.isnull(row[column]) else row[column], axis=1)
        return file
    def missing_cats(file, column, groupby_column):
        mode_values = file.groupby(groupby_column)[column].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else x.value_counts().idxmax())
        file[column] = file.apply(lambda row: mode_values[row[groupby_column]] if pd.isnull(row[column]) else row[column], axis=1)
        return file

    file = missing_num(file, 'Age', 'Survived')
    file = missing_cats(file, 'Embarked', 'Survived')
    file = missing_cats(file, 'Cabin', 'Survived')
    file = missing_num(file, 'Pclass', 'Survived')
    # add missing data
    file.to_csv("../Date/task8.csv", index=False)
    missing_values = file.isnull().sum()
    print("Missing values:")
    print(missing_values)
    print("")


def task9():
    file = pd.read_csv("../Date/train.csv")

    # extract title from the name column
    def extract_title(name):
        title_search = re.search(r'(\w+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""

    file['Title'] = file['Name'].apply(extract_title)
    def check_title_gender(row):
        title = row['Title']
        gender = row['Sex']
        if (title in ['Mr', 'Don', 'Dr', 'Sir', 'Master', 'Major', 'Col', 'Capt', 'Jonkheerta']) and (gender != 'male'):
            return False #fake male
        if (title in ['Mrs', 'Miss', 'Mlle', 'Contess', 'Lady', 'Mme']) and (gender != 'female'):
            return False  #fake female
        return True

    file['Title_Gender_Correct'] = file.apply(check_title_gender, axis=1)
    incorrect_titles = file[~file['Title_Gender_Correct']] #titles used incorrectly
    correct_titles = file[file['Title_Gender_Correct']]
    title_counts = file['Title'].value_counts() #all titles used

    plt.figure(figsize=(10, 7))
    title_counts.plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Number of people for each title')
    plt.xlabel('Title')
    plt.ylabel('Number of people')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    #printing each person with an incorrect title
    print("Incorect titles:")
    print(incorrect_titles[['Name', 'Sex', 'Title']])
    print("")

def task10():
    file = pd.read_csv("../Date/train.csv")
    file['IsAlone'] = (file['SibSp'] == 0) & (file['Parch'] == 0)

    # histrogram for chances of survival based on being alone or not
    plt.figure(figsize=(9, 6))
    sns.histplot(data=file, x='IsAlone', hue='Survived', multiple='stack', kde=False)
    plt.title('How being alone might have influenced the survival chance')
    plt.xlabel('Alone (1 = Yes, 0 = No)')
    plt.ylabel('Number of passangers')
    plt.show()

    # graphic representation on the corelation between fare, class and survival
    sns.catplot(data=file.head(100), x='Pclass', y='Fare', hue='Survived', s=25, height=6, aspect=2, marker='.', kind='swarm')
    plt.title('Corelation between fare, class and survival')
    plt.xlabel('Class')
    plt.ylabel('Fare')
    plt.show()

def main():
    while True:
        print("------Which task would you like to see?------")
        task = input()
        if task.lower() == 'exit': #if you're done checking the tasks
            break
        print("Here comes what you asked for---->\n")
        match task:
            case "task1":
                task1()
            case "task2":
                task2()
            case "task3":
                task3()
            case "task4":
                task4()
            case "task5":
                task5()
            case "task6":
                task6()
            case "task7":
                task7()
            case "task8":
                task8()
            case "task9":
                task9()
            case "task10":
                task10()
            case _:
                print(f"----Type better, no such case: {task}!-----\n")
        
if __name__ == "__main__":
    main()
