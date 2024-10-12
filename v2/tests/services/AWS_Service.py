import cv2
import boto3
import os
import tempfile
import json

class S3Uploader:
    def __init__(self, bucket_name, region_name):
        self.bucket_name = bucket_name
        self.region_name = region_name
        # Initialize the S3 client using default credentials from the environment or configuration
        self.s3_client = boto3.client('s3')

        self.sns_client = boto3.client('sns', region_name='us-east-2')
        self.topic_arn = 'arn:aws:sns:us-east-2:468348667459:SmartPictures.fifo'
        # Initialize SQS client for the FIFO queue
        self.sqs_client = boto3.client('sqs', region_name='us-east-2')

        # Replace with your actual FIFO queue URL (must end with .fifo)
        self.queue_url = 'https://sqs.us-east-2.amazonaws.com/468348667459/SmartPicturesQ.fifo'


    def upload_cv2_image(self, image, s3_key):
        """
        Upload an OpenCV image to S3.

        :param image: OpenCV image object (numpy array)
        :param s3_key: The key (path) in the S3 bucket
        """
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            local_path = tmpfile.name
            # Save the OpenCV image to the temporary file
            cv2.imwrite(local_path, image)

        try:
            # Upload the file to S3
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"Image uploaded successfully to s3://{self.bucket_name}/{s3_key}")
            # Construct the S3 object URL
            object_url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"
            return object_url
        except Exception as e:
            print(f"Error uploading to S3: {e}")
        finally:
            # Clean up the temporary file
            os.remove(local_path)

    def notify_image_processed(self, image_id, s3_url):
        message_dict = {
            'image_id': image_id,
            's3_url': s3_url,
            'status': 'processed',
            'timestamp': '2024-09-20T10:00:00Z'  # Example timestamp
        }
    
        # Convert the dictionary to a JSON string
        message_str = json.dumps(message_dict)
        
        # Deduplication ID can be the image ID (or any other unique identifier)
        deduplication_id = image_id
        
        # Message Group ID can be anything logical, like 'image-processing'
        message_group_id = 'image-processing'

        # Publish to SNS FIFO topic
        self.sns_client.publish(
            TopicArn=self.topic_arn,
            Message=message_str,
            MessageDeduplicationId=deduplication_id,  # Required for FIFO
            MessageGroupId=message_group_id,           # Required for FIFO
            Subject='New Image Processed'
        )

    def poll_sqs_fifo(self):
        while True:
            # Receive messages from the FIFO queue
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,       # FIFO processes one message at a time per MessageGroupId
                WaitTimeSeconds=10,          # Long polling to reduce empty responses
                MessageAttributeNames=['All']  # Fetch all message attributes
            )

            messages = response.get('Messages', [])
        
            for message in messages:
                # Deserialize the JSON message back into a dictionary
                message_body = json.loads(message['Body'])
                print(f"Received message: {message_body}")
                
                message_data = json.loads(message_body["Message"])
                # Example: Accessing data from the message dictionary
                image_id = message_data['image_id']
                s3_url = message_data['s3_url']
                print(f"Image ID: {image_id}, S3 URL: {s3_url}")
                
                # Process the message as needed and delete it after processing
                self.sqs_client.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
                return s3_url
