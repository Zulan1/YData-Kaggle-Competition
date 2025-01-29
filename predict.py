import pickle
from app.argparser import get_predict_args



def main():
    args = get_predict_args()

    print(f"Loading model from {args.model_path}")
    print(f"Predicting on data from {args.input_data}")
    
    model = pickle.load(open(args.model_path, 'rb'))
    df = pd.read_csv(args.input_data)
    predictions = model.predict(df)
    print(predictions)


if __name__ == '__main__':
    main()
