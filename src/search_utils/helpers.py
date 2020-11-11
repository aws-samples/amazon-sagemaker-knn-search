#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: MIT-0
import ast
from io import StringIO
import jsonlines
import json
import pickle
import boto3
import nltk
nltk.download('punkt')
nltk.download('wordnet')


def split_s3_path(s3_path):
    """
    Splits the complete s3 path to a bucket name and a key name, 
    this is useful in cases where and api requires two seperate entries (bucket and key)


    Arguments:
        s3_path {string} -- An S3 uri path

    Returns:
        bucket {string} - The bucket name
        key {string} - The key name
    """
    
    bucket = s3_path.split("/")[2]    
    key  = '/'.join(s3_path.split("/")[3:])
    
    return bucket, key

def write_dataframe_to_s3(dataframe, bucket_name, file_name, index=True, header=True):
    """
    Pushes a pandas dataframe to S3 using StringIO library

    Arguments:
        dataframe {pd.DataFrame} -- The pandas dataframe that you want to push to S3
        bucket_name {string} -- The name of the bucket in the object file path
        file_name {string} -- The name of the key/file that will show up in S3

    Keyword Arguments:
        index {bool} -- If the index of the dataframe is included (default: {True})
        header {bool} -- If the head of the dataframe is included (default: {True})
    """

    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, header=header, index=index)
    s3_resource = boto3.resource("s3")
    s3_resource.Object(bucket_name, file_name).put(Body=csv_buffer.getvalue())


def write_pickle_to_s3(obj, bucket_name, file_name):
    """
    Writes a pickle object to S3 using the pickle library and boto3 api

    Arguments:
        obj {a pickle object} -- This is the pickle object you want to push to S3
        bucket_name {string} -- The name of the bucket
        file_name {string} -- The key/file name
    """

    pickle_byte_obj = pickle.dumps(obj)
    s3_resource = boto3.resource("s3")
    s3_resource.Object(bucket_name, file_name).put(Body=pickle_byte_obj)

    
def read_pickle_from_s3(bucket_name, file_name):
    """
    Reads a pickle object from S3 using the pickle library and the boto3 api

    Arguments:
        bucket_name {string} -- The name of the bucket
        file_name {string} -- The key/file name

    Returns:
        {pickle} -- the loaded pickle object

    """
    s3_resource = boto3.resource("s3")
    pickle_obj_bytes = s3_resource.Object(bucket_name, file_name).get()["Body"].read()
    obj = pickle.loads(pickle_obj_bytes)

    return obj

def read_jsonline(fname):
    """
    Iterates over a jsonlines file and yields results

    Arguments:
        bucket_name {string} -- The name of the bucket
    """

    with jsonlines.open(fname) as reader:
        for line in reader:
            yield line

def read_jsonlines_from_s3(bucket_name, file_name):
    """
    Read jsonlines file directly from s3

    Arguments:
        bucket {string} -- The name of the bucket
        file_name {string} -- The key/file name

    Returns:
        {list} -- a list of json objects
    """

    json_lines_object = boto3.client('s3').get_object(Bucket=bucket_name, Key=file_name)

    json_lines_object_list = json_lines_object['Body'].read().decode('utf8').split('\n')[:-1]

    records = []
    for element in json_lines_object_list:
        record = ast.literal_eval(element)
        records.append(record)
        
    return records
            
def read_json_from_s3(bucket_name, file_name):
    """
    Reads a json file from S3 based on the bucket and key passed as in input

    Arguments:
        bucket_name {string} -- The name of the bucket
        file_name {string} -- The key/file name

    Returns:
        dict -- The dictionary containing the read json object 
    """
    
    s3_obj = boto3.client('s3')

    s3_clientobj = s3_obj.get_object(Bucket=bucket_name, Key=file_name)
    s3_clientdata = s3_clientobj['Body'].read().decode('utf-8')

    result=json.loads(s3_clientdata)
    
    return result


