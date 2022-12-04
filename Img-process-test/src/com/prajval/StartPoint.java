package com.prajval;

import java.io.FileNotFoundException;

import org.opencv.core.CvException;

public class StartPoint {

	public static void main(String[] args) throws FileNotFoundException {
		Start start = new Start();
		System.out.println("Start from Main");
		try {
			start.detectObjectOnImage();
		} catch (FileNotFoundException | ClassNotFoundException | NullPointerException | CvException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println(start);
		}
}
