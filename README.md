# Pytorch_model_AWS_Lambda
Deploying Pytorch models with AWS Lambda

Instructions:
install serverless	-	sudo npm install -g serverless
create folder	-	sls create --template aws-python3 --path mobilenetv2-pytorch-aws/
move into folder -	cd mobilenetv2-pytorch-aws/	
create requirements.txt	-	touch requirements.txt
install Serverless plugin serverless-python-requirements	-	sls plugin install -n serverless-python-requirements
update handler.py
create pytorch model using toch.jit.trace	-	 python model/create_mobilenet_v2_model.py 
upload model to S3	-	python model/upload_model_to_s3.py 
update serverless.yaml
deploy our function	-	npm run deploy 

URL:
[AWS Lambda REST API url]( https://1aj7jujam2.execute-api.us-east-1.amazonaws.com/dev/clasify)



Reference:
(https://towardsdatascience.com/scaling-machine-learning-from-zero-to-hero-d63796442526)