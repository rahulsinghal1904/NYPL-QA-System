from django.shortcuts import render, get_object_or_404
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.template import RequestContext
from django.views.generic.edit import FormView
from django.contrib import messages
from django.contrib.auth.models import User
from django.conf import settings
from django.db.models import Q
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from DigiSuiteClassifier.forms import ConnectCollectionForm, NewCollectionForm, UpdateCollectionForm, AnnotateCollectionForm, SelectCollectionForm, GenerateModelForm, RetrainModelForm
from pathlib import Path
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import subprocess
import logging
import datetime
from DigiSuiteClassifier.models import Vendor, VendorCollection, Collection, CollectionUpdate, TaskMonitor
from ATGen.models import CaptionData
from Users.models import VendorAuth
import os
#from django_pandas.io import read_frame
import pandas as pd
import numpy as np
import json
import pickle
import time
from DigiSuiteApp.models import Topic, TopicDetail, ContactInfo, Notification
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from DigiSuiteClassifier.scripts.collection.ChatBot import handle_chatbot_action
from DigiSuiteClassifier.scripts.collection.Chatbot_NYPL import handle_chatbot_action_test
from FaceDetect.models import FaceDetect

# Create your views here.
User = get_user_model()
logger = logging.getLogger(__name__)

def run_script(script_name, script_path, script_args, new_thread = False):
    logger.info("run ML script")
    
    # LINUX VERSION
    #venvpath = str(Path(r'/home/atgen/py383venv/bin', 'activate')) # Python venv3.8.3 # ORIGINAL
    
    # WINDOWS VERSION
    #print(settings.PYVENV_PATH)
    #venvpath_str = "r'" + settings.PYVENV_PATH + "'"
    #venvpath = str(Path(venvpath_str, 'activate'))
    #venvpath = str(Path(settings.PYVENV_PATH, 'activate'))
    
    path = str(settings.BASE_DIR) + script_path
    scriptpath = str(Path(path, script_name))
    
    logger.info("PATH: " + str(path))
    # logger.info(venvpath)
    print(f"--------------------------------- script path {scriptpath} -----------------")
    logger.info(scriptpath)
    
    arg_str = ''
    for arg in script_args:
        arg_str += '"'
        arg_str += str(arg)
        arg_str += '" '
    
    #cmd = 'source ' + venvpath + '; python ' + scriptpath + ' "' + script_args[0] + '" "' + script_args[1] + '" "' + script_args[2] + '"' # ORIGINAL
    #cmd = 'source ' + venvpath + '; python ' + scriptpath + ' "' + script_args[0] + '" "' + script_args[1] + '"'
    #cmd = 'source ' + str(venvpath) + '; python ' + str(scriptpath) + ' "' + str(script_args[0]) + '" "' + str(script_args[1]) + '"'
    #cmd = 'source ' + str(venvpath) + '; python ' + str(scriptpath) + ' ' + arg_str # LINUX VERSION - ORIGINAL WORKING
    #cmd = str(venvpath) + '; python ' + str(scriptpath) + ' ' + arg_str # WINDOWS VERSION
    #cmd = 'conda activate py37_pytorch; python ' + str(scriptpath) + ' ' + arg_str # LINUX VERSION WITH GPU (pytorch)
    # cmd = 'conda activate py37_tensorflow; python ' + str(scriptpath) + ' ' + arg_str # LINUX VERSION WITH GPU (tensorflow)
    cmd = 'python3 ' + str(scriptpath) + ' ' + arg_str

    print(f"--------- CMD :  {cmd}")
    logger.info("CMD: " + cmd)
    if(new_thread):
        subprocess.Popen("exec " + cmd, shell=True, executable='/bin/bash')
    else:
        subprocess.run(cmd, capture_output=False, shell=True, executable='/bin/bash')
    #output = subprocess.run(cmd, capture_output=True, shell=True, executable='/bin/bash', timeout=1800)
    #logger.info("SCRIPT COMPLETED with output: " + output)
    #return output

def index(request):
    notifications = Notification.objects.all()
    print("----------------notifications-------------",notifications)
    return render(request, "DigiSuiteClassifier/index.html", {'is_home_page':True, 'notifications': notifications, 'no_of_notifs':Notification.objects.count()})

@csrf_exempt
def delete_notification(request, notification_id):
    notification = get_object_or_404(Notification, pk=notification_id)
    notification.delete()
    return JsonResponse({"message": "Notification deleted"})

@csrf_exempt
@login_required
def chatbot_view(request):
    if request.method == "POST":
        # Call the chatbot logic
        response = handle_chatbot_action(request)

        # Handle errors in the response
        if "error" in response:
            return JsonResponse({"error": response["error"]}, status=400)

        # Handle valid responses
        if "answer" in response:
            # Save updated conversation chain in session
            if "conversation_chain" in response:
                request.session["conversation_chain"] = response["conversation_chain"]

            # Serialize source_documents
            source_documents = response.get("source_documents", [])
            serialized_docs = [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in source_documents
            ]

            return JsonResponse({
                "answer": response["answer"],
                "source_documents": serialized_docs  # Use serialized documents
            })

        if "message" in response:
            return JsonResponse({"message": response["message"]})

    # Render the chatbot HTML for GET requests
    return render(request, "DigiSuiteClassifier/chatbot.html")
@csrf_exempt
@login_required
def chatbot_view_test(request):
    if request.method == "POST":
        # Call the chatbot logic
        response = handle_chatbot_action_test(request)

        # Handle errors in the response
        if "error" in response:
            return JsonResponse({"error": response["error"]}, status=400)

        # Handle valid responses
        if "answer" in response:
            # Save updated conversation chain in session
            if "conversation_chain" in response:
                request.session["conversation_chain"] = response["conversation_chain"]

            # Serialize source_documents
            source_documents = response.get("source_documents", [])
            serialized_docs = [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in source_documents
            ]

            return JsonResponse({
                "answer": response["answer"],
                "source_documents": serialized_docs  # Use serialized documents
            })

        if "message" in response:
            return JsonResponse({"message": response["message"]})

    # Render the chatbot HTML for GET requests
    return render(request, "DigiSuiteClassifier/chatbot_nypl.html")

  
  
@login_required
def Home(request):
    message = "NONE"
    #full_ml_script_path = os.path.join(settings.ML_SCRIPT_PATH, 'DigiSuiteClassifier/collection')
    full_ml_script_path = os.path.join(settings.BASE_DIR, 'DigiSuiteClassifier/scripts/collection')
    
    if(VendorCollection.objects.all().count() == 0):
        #run_script('InitializeCollection.py',  full_ml_script_path, [])
        run_script('InitializeCollection.py',  '/DigiSuiteClassifier/scripts/collection', [])

    user_obj = User.objects.get(username = request.user)
    user_id = user_obj.id
    collection_obj = Collection.objects.all().order_by('id')

    page = request.GET.get('page', 1)

    paginator = Paginator(collection_obj, 10)
    try:
        paged_images = paginator.page(page)
        page = int(page)
    except PageNotAnInteger:
        paged_images = paginator.page(1)
        page = 1
    except EmptyPage:
        paged_images = paginator.page(paginator.num_pages)
    # SELECT ac.id, ac.title,at2.task_name , at2.status , at2.task_complete_time_estimate 
    # FROM atgenml_collection ac 
    # JOIN atgenml_taskmonitor at2 ON (ac.id = at2.collection_id) 
    # LEFT OUTER JOIN atgenml_taskmonitor at3 ON (ac.id = at3.collection_id 
    # AND (at2.id < at3.id ))
    # WHERE at3.id IS NULL;

    result = []
    for c in paged_images:
        t_obj = TaskMonitor.objects.filter(collection_id = c.id, user_id = user_id).last()
        if t_obj is not None:
            result.append(t_obj)
        else:
            result.append(None)
    
    serial = [i+1+((page-1)*10) for i in range(len(result))]
    form = NewCollectionForm(request.GET)

    if request.method == 'POST':
        form_filled = NewCollectionForm(request.POST)
        if(form_filled.is_valid()):
            script_args = []
            collection_selected = form_filled.cleaned_data.get('collection')
            # determine if the collection selected was already downloaded
            try:
                vcollection_obj = VendorCollection.objects.get(pk=collection_selected)
                if vcollection_obj is not None:
                    if vcollection_obj.collection_downloaded:
                        messages.success(request, f'The \'{vcollection_obj.title}\' collection was already downloaded. Consider updating the collection.')
                        return HttpResponseRedirect(reverse('home'))
                        # message = f'The \'{vcollection_obj.title}\' collection was already downloaded. Consider updating the collection.'
                        # return render(request, 'DigiSuiteClassifier/atgenml_home.html', {'message':message, 'form':form, 'collections': collection_obj, 'task_monitor':result})
                        
            except Exception as e:
                logger.info(e)
                #message = f'Invalid request: {e}'
                #return render(request, 'DigiSuiteClassifier/collection_new.html', {'message':message})
                pass

            # obtain collection uuid and title
            title = vcollection_obj.title
            uuid = vcollection_obj.uuid
            collection_shortname = vcollection_obj.collection_shortname
            vendor_id = vcollection_obj.vendor
            vendor_obj = get_object_or_404(Vendor, pk=vendor_id.id)

            # obtain user's authentication token
            try:
                vendor_auth_obj = VendorAuth.objects.get(Q(authenticated_user=request.user.id) & Q(vendor=vendor_obj.id))
            except Exception as e:
                messages.success(request, f'You do not have an Authentication Token for \'{vendor_obj.vendor_name}\' collections. Please visit Connect under Collection establish a connection.')
                return HttpResponseRedirect(reverse('home'))
                # message = f'You do not have an Authentication Token for \'{vendor_obj.vendor_name}\' collections. Please visit Connect under Collection establish a connection.'
                # return render(request, 'DigiSuiteClassifier/atgenml_home.html', {'message':message, 'form':form, 'collections': collection_obj, 'task_monitor':result})
            else:   
                auth_token = vendor_auth_obj.auth_token
                now = datetime.datetime.now()
                format = "%m_%d_%Y"
                download_date = now.strftime(format)
                
                #collection_path = '/robothon/atgen/' # change to the proper partition - create a directory for this outside of robothon
                collection_dir = collection_shortname
                collection_path = settings.TRAINING_FILES_DIR + "/" + collection_dir
                collection_images_path = settings.TRAINING_FILES_DIR + "/" + collection_dir + '/IMAGES/'
                filename = collection_path + '/' + 'dateDigitized.txt'
                                            
                try: 
                    # create directory for collection files
                    logger.info("Creating the collection directory.")
                    os.chdir(settings.TRAINING_FILES_DIR)
                    logger.info("CURRENT DIRECTORY: " + os.getcwd())
                    
                    if os.path.isdir(collection_dir):
                        logger.info(f'Directory {collection_dir} already exists.')
                    else:
                        os.mkdir(collection_dir)
                        logger.info("Collection directory was created successfully.")
                        
                        # create directory for images in the collection directory
                        if os.path.isdir(collection_dir):
                            os.chdir(collection_path)
                            if os.path.isdir("IMAGES"):
                                logger.info(f'Directory IMAGES already exists.')
                            else:
                                os.mkdir("IMAGES")
                                os.mkdir("models")
                                os.mkdir("captions")
                                logger.info("IMAGES directory was created successfully.")
                except Exception as e:
                    logger.info("Cannot create directories for collection: " + e)
                    messages.success(request, f'ERROR Please retry.')
                    return HttpResponseRedirect(reverse('home'))

                # Run NewCollection script to download selected collection
                script_args.append(filename)
                script_args.append(collection_images_path)
                script_args.append(uuid)
                script_args.append(auth_token)
                script_args.append(vcollection_obj.id)
                script_args.append(collection_dir)
                script_args.append(now)
                script_args.append(request.user)
                script_args.append(collection_path)
                output = run_script('NewCollection.py', '/DigiSuiteClassifier/scripts/collection', script_args)
                #run_script('NewCollection.py', full_ml_script_path, script_args, new_thread = True)
                
                #logger.info("New Collection script completed ", str(output))
                logger.info("New Collection script completed ")
            # message = f'Collection, \'{vcollection_obj.title}\', download started successfully.\nPlease wait for a minute to view updates in the page below.'
            messages.success(request, f'Collection, \'{vcollection_obj.title}\', download started successfully.\nPlease wait for a minute to view updates in the page below.')
            return HttpResponseRedirect(reverse('home'))

    
    return render(request, 'DigiSuiteClassifier/classifier_home.html', {'message':message, 'form':form, 'collections': paged_images, 'task_monitor':result, 'serial_no':serial, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

# -------------------------------------------------------------------------------------------------
# COLLECTION
# -------------------------------------------------------------------------------------------------
# Collection - New

@login_required
def ConnectCollection(request):
    if request.method == 'POST':
        form = ConnectCollectionForm(request.POST)
        if form.is_valid():
            collection_source_selected = form.cleaned_data.get('collection_source')
            auth_token = form.cleaned_data['auth_token']
            collection_source = Vendor.objects.get(pk=collection_source_selected)
            vendor = collection_source.vendor_name            
            try:
                validate_auth_token = VendorAuth.objects.get(Q(vendor_id=collection_source.id) & Q(authenticated_user=request.user))
                if validate_auth_token is not None:
                    resp_message = f'You already have a connection to the {vendor} collections.'
                    return render(request, 'DigiSuiteClassifier/collection_connect.html', {'message':resp_message})
            except Exception as e:
                logger.info(e)
                pass
            message = ''
            if collection_source.vendor_name == 'New York Public Library':
                try:
                    user = get_object_or_404(User, username=request.user)
                    vendor_obj = get_object_or_404(Vendor, pk=collection_source_selected)
                    VendorAuth.objects.create(authenticated_user=user,auth_token=auth_token,vendor=vendor_obj)
                    message = f'You are now connected to the {vendor} collection resources.'
                    return render(request, 'DigiSuiteClassifier/collection_connect.html', {'message':message})                    
                except Exception as e:
                    #message = e
                    message = f'An error occurred. The connection to {vendor} collection resources could not be established.'
                    return render(request, 'DigiSuiteClassifier/collection_connect.html', {'message':message})     
    else:
        form = ConnectCollectionForm()
    return render(request, 'DigiSuiteClassifier/collection_connect_form.html', {'form':form, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

@login_required
def ViewCollection(request, collection_id):
    user_obj = User.objects.get(username = request.user)
    collection_selected = Collection.objects.get(pk=collection_id)
    t_obj = TaskMonitor.objects.filter(collection_id = collection_selected.id, user_id = user_obj.id).last()
    collection_path  = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/IMAGES" 

    data_path = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/captions/" # ON SERVER
    data_file = str(Path(data_path, 'description_details.txt'))
    
    print("path check", data_path, collection_path)
    east_dict = {}
    east_result_file = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/east_result.csv"
    if(os.path.exists(east_result_file)):
        east_df = pd.read_csv(east_result_file)
        east_dict = east_df.set_index('filename').to_dict()['east_decision']
    print("read eastresukt")
    alt_text_dict = {}
    alt_text_file = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/temp.txt"
    if(os.path.exists(east_result_file) and os.path.exists(alt_text_file)):
        print("inside if block")
        alt_text_df = pd.read_csv(alt_text_file, sep=",", header=None)
        alt_text_df = alt_text_df.iloc[:, [0, -1]]
        print(alt_text_df)
        alt_text_dict = alt_text_df.set_index(0).to_dict()[4]

    print(collection_selected)
    print(collection_id)
    print("read temp.txt")
    if(os.path.exists(data_file)):
        df = pd.read_csv(data_file, sep=",", header=None)
        df.columns = ["image_id", "original_confidence_score", "new_confidence_score", "perplexity", "alt_text"]
        df["image_path"] = df["image_id"] + ".jpg"
        df.index.names = ['id']
        df["alt_text"] = df["alt_text"].str.rstrip()
        df["alt_text"] = df["alt_text"].str.lstrip()
        
        

    else:
        print("data file doesn't exist")
        captions = os.listdir(collection_path)
        df = pd.DataFrame(captions)
        df.columns = ["image_path"]
        df["image_id"] = df["image_path"].str.replace(".jpg", "")
        df["alt_text"] = [alt_text_dict[key] if key in alt_text_dict else None for key in df['image_id']]
        df.index.names = ['id']

    df["east_decision"] = [east_dict[key] if key in east_dict else None for key in df['image_path']]

    print(east_dict)

    inactive_captions = []
    json_records = df.reset_index().to_json(orient='records')
    captions = []
    captions = json.loads(json_records)
    
    if len(captions)>9:
        inactive_captions = captions[9:]
        captions = captions[:9]

    # logger.critical(captions)
    images_root_path = settings.IMAGE_ROOT
    return render(request, 'DigiSuiteClassifier/view_collection.html', {'obj':collection_selected, 'first':captions, 'inactive':inactive_captions, 'curr_task':t_obj, 'images_root_path':images_root_path, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

@login_required
def ViewCollectionTable(request):
    user_obj = User.objects.get(username = request.user)
    collection_id = request.POST.get("collection_id", 1)
    collection_selected = Collection.objects.get(pk=collection_id)
    t_obj = TaskMonitor.objects.filter(collection_id = collection_selected.id, user_id = user_obj.id).last()
    collection_path  = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/IMAGES" 

    data_path = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/captions/" # ON SERVER
    data_file = str(Path(data_path, 'description_details.txt'))
    
    east_dict = {}
    east_result_file = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/east_result.csv"
    if(os.path.exists(east_result_file)):
        east_df = pd.read_csv(east_result_file)
        east_dict = east_df.set_index('filename').to_dict()['east_decision']

    tags_dict = {}
    tags_result_file = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/tags.csv"
    if(os.path.exists(tags_result_file)):
        tags_df = pd.read_csv(tags_result_file)
        tags_dict = tags_df.set_index('key').to_dict()['tags']
    
    alt_text_dict = {}
    alt_text_file = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/temp.txt"
    if(os.path.exists(alt_text_file)):
        alt_text_df = pd.read_csv(alt_text_file, sep=",",  header=None)
        alt_text_df = alt_text_df.iloc[:, [0, -1]]
        alt_text_dict = alt_text_df.set_index(0).to_dict()[4]

    if(os.path.exists(data_file)):
        df = pd.read_csv(data_file, sep=",", header=None)
        df.columns = ["image_id", "original_confidence_score", "new_confidence_score", "perplexity", "alt_text"]
        df["image_path"] = df["image_id"] + ".jpg"
        df.index.names = ['id']
        df["alt_text"] = df["alt_text"].str.rstrip()
        df["alt_text"] = df["alt_text"].str.lstrip()
        
        

    else:
        captions = os.listdir(collection_path)
        df = pd.DataFrame(captions)
        df.columns = ["image_path"]
        df["image_id"] = df["image_path"].str.replace(".jpg", "")
        df["alt_text"] = [alt_text_dict[key] if key in alt_text_dict else None for key in df['image_id']]
        df.index.names = ['id']

    df["east_decision"] = [east_dict[key] if key in east_dict else None for key in df['image_path']]
    df["tags"] = [tags_dict[key] if key in tags_dict else None for key in df['image_id']]
    # print(df["east_decision"])
    print(df["tags"])
    
    json_records = df.reset_index().to_json(orient='records')
    captions = []
    captions = json.loads(json_records)
    
    face_detect_data = FaceDetect.objects.filter(collection=collection_selected)
    
    page = request.POST.get('page', 1)
    # logger.critical(request.POST)
    # paged_images = images
    paginator = Paginator(captions, 20)
    try:
        paged_images = paginator.page(page)
        page = int(page)
    except PageNotAnInteger:
        paged_images = paginator.page(1)
        page = 1
    except EmptyPage:
        paged_images = paginator.page(paginator.num_pages)

    serial = [i+1+((page-1)*20) for i in range(len(paged_images))]

    # logger.critical(captions)
    images_root_path = settings.IMAGE_ROOT
    return render(request, 'DigiSuiteClassifier/view_collection_table.html', {'collection_obj':collection_selected, 'facedetect': face_detect_data, 'first':paged_images, 'serial':serial, 'images_root_path':images_root_path, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})


@login_required
def UpdateSingleCaption(request):
    if request.method == 'POST':
        # logger.critical(request.POST)
        image_id_req = request.POST.get("image_id", -1)
        collection_id = request.POST.get('collection_id', -1)
        new_alt_text = request.POST.get("alt_text", "")
        collection_selected = Collection.objects.get(pk=collection_id)

        data_path = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/captions/" # ON SERVER
        data_file = str(Path(data_path, 'description_details.txt'))
        df = pd.read_csv(data_file, sep=",", header=None)
        df.columns = ["image_id", "original_confidence_score", "new_confidence_score", "perplexity", "alt_text"]
        df["alt_text"] = df["alt_text"].str.rstrip()
        df["alt_text"] = df["alt_text"].str.lstrip()
        df.index.names = ['id']

        df.loc[df['image_id']== image_id_req, "alt_text"] = new_alt_text
        df = df[["image_id", "original_confidence_score", "new_confidence_score", "perplexity", "alt_text"]]
        df.to_csv(data_file, sep=",",header=False, index=False)
        
        return HttpResponse("Done")



@login_required
def CustomCollection(request):
    form = SelectCollectionForm(request.GET)
    paged_images=[]
    collection_selected=None
    if request.method == 'POST':
        wtd = request.POST.get('WTD', False)
        if(wtd):
            imagess = request.POST.get('imagess')
            imagess = imagess.split("\n")
            collection_name = request.POST.get('user_collection_name')
            imagess2 = []

            for i in range(0, len(imagess)):
                a = imagess[i]
                if(a=="" or (not a[0].isalnum())):
                    continue
                temp = a.split("-")
                for j in range(0, len(temp)):
                    temp[j] = temp[j].strip()
                imagess2.append(temp)
            
            user = get_object_or_404(User, username=request.user)
            short = str(user)
            # print(title)
            temp = short
            short = ""
            for i in temp:
                short+= i.upper()
            
            short += "_"
            short_ = collection_name.split(" ")

            for s in short_:
                if (len(s) >0 and s[0].isalpha()):
                    short+=s[0].upper()

            collection_dir = short
            collection_path = settings.TRAINING_FILES_DIR + "/" + collection_dir
            collection_images_path = settings.TRAINING_FILES_DIR + "/" + collection_dir + '/IMAGES/'
            filename = collection_path + '/' + 'dateDigitized.txt'
            images_list = collection_path + "/images.pkl"
                                        
            try: 
                # create directory for collection files
                logger.info("Creating the collection directory.")
                os.chdir(settings.TRAINING_FILES_DIR)
                logger.info("CURRENT DIRECTORY: " + os.getcwd())
                
                if os.path.isdir(collection_dir):
                    logger.info(f'Directory {collection_dir} already exists.')
                else:
                    os.mkdir(collection_dir)
                    logger.info("Collection directory was created successfully.")
                    
                    # create directory for images in the collection directory
                    if os.path.isdir(collection_dir):
                        os.chdir(collection_path)
                        if os.path.isdir("IMAGES"):
                            logger.info(f'Directory IMAGES already exists.')
                        else:
                            os.mkdir("IMAGES")
                            os.mkdir("models")
                            os.mkdir("captions")
                            logger.info("IMAGES directory was created successfully.")
            except Exception as e:
                logger.info("Cannot create directories for collection: " + e)
                messages.success(request, f'ERROR Please retry.')
                return HttpResponseRedirect(reverse('custom-collection'))
            
            with open(images_list, 'wb') as f:
                pickle.dump(imagess2, f)

            script_args= []
            script_args.append(filename)
            script_args.append(collection_images_path)
            script_args.append(collection_path)
            script_args.append(collection_dir)
            script_args.append(images_list)
            script_args.append(request.user)
            script_args.append(collection_name)
            script_args.append(settings.TRAINING_FILES_DIR)

            run_script('NewUserCollection.py', '/DigiSuiteClassifier/scripts/collection', script_args, new_thread = True)
            messages.success(request, 'Custom collection created. Give 5-10 mins to complete the process.')
            return HttpResponseRedirect(reverse('custom-collection'))

        else:
            page = request.POST.get('page', 1)
            # logger.critical(request.POST)
            collection_id = request.POST.get('collection_id', -1)
            # p_form = SelectCollectionForm(request.POST)
            # collection_id = p_form.cleaned_data.get('collection')
            collection_selected = Collection.objects.get(pk=collection_id)
            collection_path  = f"{settings.TRAINING_FILES_DIR}/{collection_selected.collection_directory_name}/IMAGES" 
            # collection_path  = f"{settings.TRAINING_FILES_DIR}/NYPL_KD/IMAGES" 
            images = os.listdir(collection_path)
            # paged_images = images
            paginator = Paginator(images, 20)
            try:
                paged_images = paginator.page(page)
                page = int(page)
            except PageNotAnInteger:
                paged_images = paginator.page(1)
                page = 1
            except EmptyPage:
                paged_images = paginator.page(paginator.num_pages)

            serial = [i+1+((page-1)*20) for i in range(len(paged_images))]
            return render(request,"DigiSuiteClassifier/custom_collection_table.html", {'images': paged_images, 'collection_obj':collection_selected, 'serial':serial})
                
    return render(request, 'DigiSuiteClassifier/custom_collection.html', {'collection_obj':collection_selected, 'form':form, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})


@login_required
def NewCollection(request):
    if request.method == 'POST':
        form = NewCollectionForm(request.POST)
        if form.is_valid():
            message = ''
            script_args = []
            collection_selected = form.cleaned_data.get('collection')
            # determine if the collection selected was already downloaded
            try:
                vcollection_obj = VendorCollection.objects.get(pk=collection_selected)
                if vcollection_obj is not None:
                    if vcollection_obj.collection_downloaded:
                        message = f'The \'{vcollection_obj.title}\' collection was already downloaded. Consider updating the collection.'
                        return render(request, 'DigiSuiteClassifier/collection_new.html', {'message':message})
            except Exception as e:
                logger.info(e)
                #message = f'Invalid request: {e}'
                #return render(request, 'DigiSuiteClassifier/collection_new.html', {'message':message})
                pass

            # obtain collection uuid and title
            title = vcollection_obj.title
            uuid = vcollection_obj.uuid
            collection_shortname = vcollection_obj.collection_shortname
            vendor_id = vcollection_obj.vendor
            vendor_obj = get_object_or_404(Vendor, pk=vendor_id.id)

            # obtain user's authentication token
            try:
                vendor_auth_obj = VendorAuth.objects.get(Q(authenticated_user=request.user.id) & Q(vendor=vendor_obj.id))
            except Exception as e:
                message = f'You do not have an Authentication Token for \'{vendor_obj.vendor_name}\' collections. Please visit Connect under Collection establish a connection.'
                return render(request, 'DigiSuiteClassifier/collection_new.html', {'message':message})
            else:   
                auth_token = vendor_auth_obj.auth_token
                now = datetime.datetime.now()
                format = "%m_%d_%Y"
                download_date = now.strftime(format)
                
                #collection_path = '/robothon/atgen/' # change to the proper partition - create a directory for this outside of robothon
                collection_dir = collection_shortname + "_initial_" + download_date
                collection_path = settings.TRAINING_FILES_DIR + "/" + collection_dir
                collection_images_path = settings.TRAINING_FILES_DIR + "/" + collection_dir + '/IMAGES/'
                filename = collection_path + '/' + 'dateDigitized1.txt'
                                            
                try: 
                    # create directory for collection files
                    logger.info("Creating the collection directory.")
                    os.chdir(settings.TRAINING_FILES_DIR)
                    logger.info("CURRENT DIRECTORY: " + os.getcwd())
                    
                    if os.path.isdir(collection_dir):
                        logger.info(f'Directory {collection_dir} already exists.')
                    else:
                        os.mkdir(collection_dir)
                        logger.info("Collection directory was created successfully.")
                        
                        # create directory for images in the collection directory
                        if os.path.isdir(collection_dir):
                            os.chdir(collection_path)
                            if os.path.isdir("IMAGES"):
                                logger.info(f'Directory IMAGES already exists.')
                            else:
                                os.mkdir("IMAGES")
                                logger.info("IMAGES directory was created successfully.")
                except Exception as e:
                    logger.info("Cannot create directories for collection: " + e)

                # Run NewCollection script to download selected collection
                script_args.append(filename)
                script_args.append(collection_images_path)
                script_args.append(uuid)
                script_args.append(auth_token)
                script_args.append(vcollection_obj.id)
                script_args.append(collection_dir)
                script_args.append(now)
                script_args.append(request.user)
                #output = run_script('NewCollection.py', '/DigiSuiteClassifier/scripts/collection', script_args)
                run_script('NewCollection.py', '/DigiSuiteClassifier/scripts/collection', script_args, new_thread = True)
                #logger.info("New Collection script completed ", str(output))
                logger.info("New Collection script completed ")
                
                # create entry in collection table of newly downloaded collection, and flag as downloaded because should be downloaded once                            
                #Collection.objects.create(title=title,collection_directory_name=collection_dir,date_downloaded=now,collection_downloaded=True,vendor_collection=vendor_obj)
                # Collection.objects.create(title=title,collection_directory_name=collection_dir,date_downloaded=now,vendor_collection=vcollection_obj)
                # vcollection_obj.collection_downloaded = True
                # vcollection_obj.save()
                #message = f'Collection download completed successfully: {output}'
                message = f'Collection, \'{vcollection_obj.title}\', download completed successfully'
                return render(request, 'DigiSuiteClassifier/collection_new.html', {'message':message})
    else:
        if(VendorCollection.objects.all().count() == 0):
            run_script('InitializeCollection.py',  '/DigiSuiteClassifier/scripts/collection', [])
        form = NewCollectionForm()
    return render(request, 'DigiSuiteClassifier/collection_new_form.html', {'form':form, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

# -------------------------------------------------------------------------------------------------
# Collection - Update

# Check if all new images from UpdateCollection have 
# alt-text entered by librarian - return true
# else, return false - this prevents the generation
# of a new internal model on an updated collection
# if there are newly downloaded images that do not
# contain alt-text
def VerifyNewImagesContainAltText():
    pass

@login_required
def UpdateCollection(request):
    if request.method == 'POST':
        form = UpdateCollectionForm(request.POST)
        if form.is_valid():
            message = ''
            script_args = []
            collection_selected = form.cleaned_data.get('collection')
            
            try: # determine if the collection selected was already downloaded
                vcollection_obj = VendorCollection.objects.get(pk=collection_selected)
                if vcollection_obj is not None:
                    if not vcollection_obj.collection_downloaded:
                        message = f'The \'{vcollection_obj.title}\' collection was not downloaded. Please download this collection before updating it.'
                        return render(request, 'DigiSuiteClassifier/collection_new.html', {'message':message})
            except Exception as e:
                message = f'Collection does not exist.'
                return render(request, 'DigiSuiteClassifier/collection_update.html', {'message':message})
            else:
                # obtain collection uuid and title
                title = vcollection_obj.title
                uuid = vcollection_obj.uuid
                collection_shortname = vcollection_obj.collection_shortname
                vendor_id = vcollection_obj.vendor
                vendor_obj = get_object_or_404(Vendor, pk=vendor_id.id)

                try: # obtain user's authentication token
                    vendor_auth_obj = VendorAuth.objects.get(Q(authenticated_user=request.user.id) & Q(vendor=vendor_obj.id))
                except Exception as e:
                    message = f'You do not have an Authentication Token for \'{vendor_obj.vendor_name}\' collections. Please visit Connect under Collection establish a connection.'
                    return render(request, 'DigiSuiteClassifier/collection_new.html', {'message':message})
                else:   
                    auth_token = vendor_auth_obj.auth_token
                    now = datetime.datetime.now()
                    format = "%m_%d_%Y"
                    download_date = now.strftime(format)
                
                    print("----------------------- collection info")
                    print(str(vcollection_obj))
                    print(str(title))
                    print(str(uuid))
                    print(str(collection_shortname))
                    print(str(vendor_id))
                    print(request.user)
                    print(auth_token)
                
                # obtain user's authentication token
                vendor_obj = get_object_or_404(Vendor, pk=vendor_id.id)
                
                try:
                    initial_collection = Collection.objects.get(pk=vendor_obj.id)


                    now = datetime.datetime.now()
                    print(str(now))
                    format = "%m_%d_%Y"
                    download_date = now.strftime(format)
                    print(str(download_date))
                    
                    #collection_path = '/robothon/atgen/' # change to the proper partition - create a directory for this outside of robothon
                    collection_dir = collection_shortname + "_update_" + download_date
                    collection_path = settings.TRAINING_FILES_DIR + collection_dir
                    collection_images_path = settings.TRAINING_FILES_DIR + collection_dir + '/IMAGES'
                    filename = collection_path + '/' + 'dateDigitized1.txt'
                    
                    print(collection_dir)
                    print(collection_path)
                    print(collection_images_path)
                    print(filename)

                    # Run UpdateCollection script to download selected collection
                    script_args.append(filename)
                    script_args.append(collection_images_path)
                    script_args.append(uuid)
                    script_args.append(auth_token)
                    #output = run_script('UpdateCollection.py', '/DigiSuiteClassifier/scripts/collection', script_args)
                    output = "completed"
                    logger.info("Update Collection script completed ", str(output))        
                    message = f'Collection was updated successfully: {output}'

                    # create entry in collection table of newly downloaded collection, and flag as downloaded because should be downloaded once                            
                    CollectionUpdate.objects.create(title=title,collection_directory_name=collection_dir,date_updated=now,collection=initial_collection,vendor_collection=vcollection_obj)
                    message = 'Collection update completed: ' # + output
                    return render(request, 'DigiSuiteClassifier/collection_update.html', {'message':message})

                except Exception as e:
                    message = f'Collection \'{vendor_obj.vendor_name}\' does not exist. Please download this collection before updating it.'
                    return render(request, 'DigiSuiteClassifier/collection_update.html', {'message':message})



    else:
        form = UpdateCollectionForm()
    return render(request, 'DigiSuiteClassifier/collection_update_form.html', {'form':form, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

# -------------------------------------------------------------------------------------------------
# Collection - Annotate

@login_required
def AnnotateCollection(request):
    if request.method == 'POST':
        form = AnnotateCollectionForm(request.POST)
        if form.is_valid():
            collection_selected = form.cleaned_data.get('collection')
            captions = CaptionData.objects.all()
            
            #EXIFTOOL = '/home/metacomp/CadillacDB4/tools/exiftool/exiftool'
            #print(str(settings.BASE_DIR))
 
            #images_path = os.path.join(settings.BASE_DIR, 'atgenml\scripts\trainingFiles\IMAGES')
            #images_dir = str(Path(settings.BASE_DIR, 'DigiSuiteClassifier/scripts/trainingFiles/IMAGES/')) # ORIGINAL (local)
            #images_dir = str(Path(settings.BASE_DIR, 'static/images/trainingImages/FSA/'))
            
            #images_dir = str(settings.BASE_DIR) + '/DigiSuiteClassifier/scripts/trainingFiles/IMAGES' # ON SERVER (small sample)
            images_dir = '/robothon/atgen/trainingFiles/NYPL_FSA_initial_04_30_2021' # ON SERVER (full collection)
            #image_loc = str(Path(imageFiles_path, request.FILES['image_file'].name))
            
            
            tools_path = os.path.join(settings.BASE_DIR, "tools/exiftool/")
            #print(str(tools_path))
            print(str(images_dir))
            #exiftool = str(Path(tools_path, 'exiftool.exe')) # ORIGINAL (WINDOWS VERSION)
            exiftool = str(Path(tools_path, 'exiftool')) # ON SERVER
            #exiftool = '/home/atgen/ATGen/tools/exiftool/exiftool'
            #print(str(exiftool))
            
            # for cap in captions:
                # fs = FileSystemStorage(location=images_dir)
                # img_filename = cap.image_id + ".jpg"
                # img_file_url = images_dir + "\\" + fs.url(img_filename)
                # alt_tag = cap.alt_text
                # alt_tag_list = alt_tag.split(' ')
                # for item in alt_tag_list:
                    # item = item.strip()
                    # print(exiftool + " -keywords+=" + item + " " + images_dir + "\\" + fs.url(img_filename)[1:])
                    # os.system(exiftool + " -keywords+=" + item + " " + images_dir + "\\" + fs.url(img_filename)[1:])
            
            for cap in captions:
                fs = FileSystemStorage(location=images_dir)
                img_filename = cap.image_id + ".jpg"
                img_file_url = images_dir + "\\" + fs.url(img_filename)
                alt_tag = cap.alt_text
                alt_tag_list = alt_tag.split(' ')
                
                item = cap.alt_text
                item = item.strip()
                # print(exiftool + ' -keywords+="' + item + '" ' + images_dir + "\\" + fs.url(img_filename)[1:]) # ORIGINAL (local)
                # os.system(exiftool + ' -keywords+="' + item + '" ' + images_dir + "\\" + fs.url(img_filename)[1:]) # ORIGINAL (local)                
                
                print(exiftool + ' -keywords+="' + item + '" ' + images_dir + "/" + fs.url(img_filename)[1:]) # ON SERVER
                os.system(exiftool + ' -keywords+="' + item + '" ' + images_dir + "/" + fs.url(img_filename)[1:]) # ON SERVER 
            
                #cmd = exiftool + ' -keywords+="' + item + '" ' + images_dir + "\\" + fs.url(img_filename)[1:]
                #print(cmd)
                #logger.info("CMD: " + cmd)
                #output = subprocess.run(cmd, timeout=1800)
                #output = subprocess.run(cmd, capture_output=True, shell=True, executable='/bin/bash', timeout=1800)
                #print(output)
            message = 'The collection was annotated successfully.'
            return render(request, 'DigiSuiteClassifier/msg_success.html', {'message':message})
    else:
        form = AnnotateCollectionForm()
    return render(request, 'DigiSuiteClassifier/collection_annotate_form.html', {'form': form, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

# -------------------------------------------------------------------------------------------------
# Collection - Export

@login_required
def ExportCollection(request):
    if request.user.is_authenticated:
        print("REACHED EXPORT COLLECTION - TEST DB ACCESS FROM SCRIPTS")
        message = "The Export Collection test script was started"
        
        script_args = []
        script_args.append("arg_1")
        script_args.append("arg_2")
        script_args.append(request.user)
        run_script('ExportCollection.py', '/DigiSuiteClassifier/scripts/collection', script_args)
        
    return render(request, 'DigiSuiteClassifier/msg_success.html', {'message':message})

# -------------------------------------------------------------------------------------------------
# Collection - Monitor

@login_required
def CollectionMonitor(request):
    if request.user.is_authenticated:
        newcollectiontasks = TaskMonitor.objects.filter(Q(task_category="COLLECTION") & Q(task_name="NEW_COLLECTION"))
        updatecollectiontasks = TaskMonitor.objects.filter(Q(task_category="COLLECTION") & Q(task_name="UPDATE_COLLECTION"))
        return render(request, 'DigiSuiteClassifier/collection_monitor.html', {'newcollectiontasks':newcollectiontasks, 'updatecollectiontasks':updatecollectiontasks, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})


# -------------------------------------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------------------------------------
@login_required
def GenerateModels(request):
    print("-------------- REQUEST -------------")
    if request.method == 'POST':
        print("---------------------- POST REQUEST ----------")
        #print(request.POST.get('collection_id', -1))
        # Get the dir path of the collection for running east model
        print(request.POST)
        
        collection_id = int(request.POST.get('collection', -1))
        
        if (collection_id == -1):
            collection_id = int(request.POST.get('collection_id', -1))

        # collection_id = int(request.POST.get('collection', -1))
        # collection_id = 8
        print("----- Collection ID = " + str(collection_id))
        print(os.getcwd())
        collection_selected = Collection.objects.get(pk=collection_id)
        collection_images_path = settings.TRAINING_FILES_DIR + '/' + collection_selected.collection_directory_name + '/IMAGES'
        collection_dir_path = settings.TRAINING_FILES_DIR + '/' + collection_selected.collection_directory_name
        # pb file path only for east model
        pbfilepath ='/../DigiSuiteScripts/DigiSuiteClassifier/east/frozen_east_text_detection.pb'
        script_args_east = []
        script_args_east.append('-east')
        script_args_east.append(pbfilepath)
        script_args_east.append('-i')
        script_args_east.append(collection_images_path)
        script_args_east.append('-notif')
        script_args_east.append(Notification)

    
        # Run East Model to classify the images at local
        print('---------------Running East Model---------------')
        # run_script('EastModel.py', '/../DigiSuiteScripts/DigiSuiteClassifier/east', script_args_east, new_thread = True)
        # time.sleep(1)

        # cmd = 'python3 ' + str(scriptpath) + ' -east ' + pbfilepath + ' -i ' + script_args + '/IMAGES'

        # Get the dir path for running the atgen model(bootstrap model)
        script_args= []
        script_args.append(collection_dir_path)
        script_args.append(collection_selected.collection_directory_name)
        script_args.append(collection_id)
        script_args.append(request.user)
        script_args.append(script_args_east)

        # run boostrap model on hpc
        print("------------- Start to Run Model on HPC -------------")
        # run_script('GenerateModels.py', '/ATGen/scripts/hpc', script_args, new_thread = True)
        run_script('GenerateModels.py', '/../DigiSuiteScripts/DigiSuiteClassifier/hpc', script_args, new_thread = True)
        
        messages.success(request, 'Model Training Started')
        return HttpResponseRedirect(reverse('view-collection', kwargs={'collection_id':collection_id}))
    else:
        form = GenerateModelForm()
    return render(request, 'DigiSuiteClassifier/model_generate_form.html', {'form': form, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

@login_required
def RetrainModels(request):
    print("I am here in retrain")
    print(f"--------- request.method == {request.method}")
    if request.method == 'POST':
        print("--------------------- request.POST ----------")
        # print(request.POST["collection"])
        collection_id = int(request.POST.get('collection', -1))
        
        if (collection_id == -1):
            collection_id = int(request.POST.get('collection_id', -1))

        print(f"---------collection_id ====== {collection_id}")
        collection_selected = Collection.objects.get(pk=collection_id)

        print(f"------ collection_selected = {collection_selected}")

        collection_dir_path = settings.TRAINING_FILES_DIR + '/' + collection_selected.collection_directory_name
        print(f"------ collection_dir_path = {collection_dir_path}")

        script_args= []
        script_args.append(collection_dir_path)
        script_args.append(collection_selected.collection_directory_name)
        script_args.append(collection_id)
        script_args.append(request.user)

        run_script('RetrainModels.py', '/../DigiSuiteScripts/DigiSuiteClassifier/hpc', script_args, new_thread = True)
        
        # script_args.append(collection_selected.collection_directory_name)
        # script_args.append(collection_id)
        # script_args.append(collection_dir_path)
        # run_script('GetUpdates.py', '/DigiSuiteClassifier/scripts/hpc', script_args, new_thread = True)
        messages.success(request, 'Model is being retrained')
        return HttpResponseRedirect(reverse('view-collection', kwargs={'collection_id':collection_id}))
    else:
        form = RetrainModelForm()
    return render(request, 'DigiSuiteClassifier/model_retrain_form.html', {'form': form, 'notifications': Notification.objects.all(), 'no_of_notifs':Notification.objects.count()})

# HTTP Error 400
def bad_request(request, Exception):
    response = render('400.html', context_instance=RequestContext(request))
    #response = render_to_response('400.html', context_instance=RequestContext(request))
    response.status_code = 400
    return response

# HTTP Error 403
def permission_denied(request, Exception):
    response = render('403.html', context_instance=RequestContext(request))
    #response = render_to_response('403.html', context_instance=RequestContext(request))
    response.status_code = 403
    return response

# HTTP Error 404
def page_not_found(request, Exception):
    response = render('404.html', context_instance=RequestContext(request))
    #response = render_to_response('404.html', context_instance=RequestContext(request))
    response.status_code = 404
    return response

# HTTP Error 500
def server_error(request):
    response = render('500.html', context_instance=RequestContext(request))
    #response = render_to_response('500.html', context_instance=RequestContext(request))
    response.status_code = 500
    return response
