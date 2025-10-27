



def handler_fit(event, _):
    model_parametrisation = event["model_parametrisation"]
    [your code here]
    return {
        "statusCode": "200",
        "body": json.dumps(
            {"model_path": [your_model_stored_path]
             }
             ),
    }



if __name__ == "__main__":
    main()
