import pickle
from app.argparser import get_predict_args



def main():
    args = get_predict_args()

    print(f"Loading model from {args.model_path}")
    print(f"Predicting on data from {args.input_data}")
    
    imputer = pickle.load(open(args.imputer_path, 'rb'))
    model = pickle.load(open(args.model_path, 'rb'))
    df = pd.read_csv(args.input_data)
    df = imputer.transform(df)
    predictions = model.predict(df)


if __name__ == '__main__':
    main()
