package com.prajval;

import java.io.FileNotFoundException;

public class StartPoint {

	public static void main(String[] args) throws FileNotFoundException {
		Start start = new Start();
		System.out.println("Start from Main");
		start.detectObjectOnImage();
		System.out.println(start);
		}
}
