# Image-Processing

First Task to Process the Image and Identify the Number from That Image and Save that Number in Memory.

This Project Will Detect The Objects Automatically,
After Identifing It will Crop and Save Them Auto.

      First we load DDl file From OpenCV
			nu.pattern.OpenCV.loadLocally();
			String DDLPath = "C:\\opencv\\build\\java\\x64\\opencv_java460.dll";
			System.load(DDLPath);     
      
      Then load the COCO class labels our YOLO model was trained on
		    Scanner scan = new Scanner(new FileReader("E:\\Img Detection cfg files\\darknet\\data\\coco.names"));
		    
		 
      load our YOLO object detector trained on COCO dataset
		    String LoadCfgFile = "C:\\Users\\PRAJVAL BHALE\\Documents\\Yolo files\\yolov4.cfg"; 
		    String LoadWeightFile = "C:\\Users\\PRAJVAL BHALE\\Documents\\Yolo files\\yolov4.weights";
		    System.out.println("YOLO file's Loaded");
		    System.out.println("before the Dnn");
		    Net dnnNet = Dnn.readNet(LoadWeightFile, LoadCfgFile);
		    
		    // YOLO on GPU:
		       dnnNet.setPreferableBackend(Dnn.DNN_BACKEND_CUDA);
		       dnnNet.setPreferableTarget(Dnn.DNN_TARGET_CUDA);
		        
		       // generate radnom color in order to draw bounding boxes
		       Random random = new Random();
		        ArrayList<Scalar> colors = new ArrayList<Scalar>();
		        
		        
    		    load our input image
		        Mat img = Imgcodecs.imread("E:\\prajwal\\11.jpg", Imgcodecs.IMREAD_COLOR); // dining_table.jpg soccer.jpg baggage_claim.jpg
		        
            Determine the output layer names that we need from YOLO, The forward() function in OpenCV’s Net class needs the ending 
            layer till which it should run in the network.
		        getUnconnectedOutLayers() vraca indexe za: yolo_82, yolo_94, yolo_106, (index su 82, 94 i 106)
            u network u:
            
		        List<String> layerNames = dnnNet.getLayerNames();
		        List<String> outputLayers = new ArrayList<String>();
		        for (Integer i : dnnNet.getUnconnectedOutLayers().toList()) 
		        {
		            outputLayers.add(layerNames.get(i - 1));
		        }
		             
		        Now, do so-called “non-maxima suppression” Non-maximum suppression is performed on the boxes whose confidence is equal to or greater than the threshold.
		        This will reduce the number of overlapping boxes:
		        
            MatOfInt indices =  getBBoxIndicesFromNonMaximumSuppression(boxes,confidences);
		        
            Finally, go over indices in order to draw bounding boxes on the image:
		        img =  drawBoxesOnTheImage(img,
		                                   indices,
		                                   boxes,
		                                   cocoLabels,
		                                   class_ids,
		                                   colors);
		        HighGui.imshow("Test", img );    
		        HighGui.waitKey(10000);     
	
				We need to prepare some data structure  in order to store the data returned by the network  (ie, after Net.forward() call)) So, Initialize our lists of
        detected bounding boxes, confidences, and  class IDs, respectively, This is what this method will return:
				
        HashMap<String, List> result = new HashMap<String, List>();
				result.put("boxes", new ArrayList<Rect2d>());
				result.put("confidences", new ArrayList<Float>());
				result.put("class_ids", new ArrayList<Integer>());

				The input image to a neural network needs to be in a certain format called a blob.
				In this process, it scales the image pixel values to a target range of 0 to 1 using a scale factor of 1/255.
				It also resizes the image to the given size of (416, 416) without cropping
				Construct a blob from the input image and then perform a forward  pass of the YOLO object detector, giving us our bounding boxes and  associated probabilities:

				Mat blob_from_image = Dnn.blobFromImage(img, 1 / 255.0, new Size(416, 416), // Here we supply the spatial size that the Convolutional Neural Network expects.
				new Scalar(new double[]{0.0, 0.0, 0.0}), true, false);
				dnnNet.setInput(blob_from_image);

				The output from network's forward() method will contain a List of OpenCV Mat object, so lets prepare one
				List<Mat> outputs = new ArrayList<Mat>();

				Finally, let pass forward through network. The main work is done here:  
				dnnNet.forward(outputs, outputLayers);

				 Each output of the network outs (i.e, each row of the Mat from 'outputs') is represented by a vector of the number of classes + 5 elements.  The first 4 
         elements represent center_x, center_y, width and height.
				 The fifth element represents the confidence that the bounding box encloses the object.
				 The remaining elements are the confidence levels (ie object types) associated with each class.
				 The box is assigned to the category corresponding to the highest score of the box:
				
				Rect2d box = null;
				for(Mat output : outputs) 
				{
					This loop over each of the detections. Each row is a candidate detection,
					System.out.println("Output.rows(): " + output.rows() + ", Output.cols(): " + output.cols());
						
					
					This will save crop image
				    Mat markedImage = new Mat();
				    Imgcodecs.imwrite("E:\\output\\crop2.png",markedImage);
				    System.out.println("Crop File Saved to This Path :-- E:\\output\\crop1.jpg");
					
				    This Will Saving the output image
				    Imgcodecs.imwrite("E:\\output\\out.jpg", img);
				    System.out.println("Output Saved");
				
		Returns index of maximum element in the list 
    
    
