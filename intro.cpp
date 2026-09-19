#include <iostream>
#include <string>
// header file library that lets us work with input and output objects
// e.g. cout
using namespace std;
//means we can use names for objects and variables from the std lib
/*
int main()
{
  cout << "Hello world!\n";
  cout << "Have a good day!\n";
// cout is an object used together with the inseetion operator <<
// to output/print text. 
  return 0;
}
*/

// can also omit using namespace std; by using this:
// std::cout << "Hello World!\n"
// but using 'using namespace std' makes the code shorter

// can also use cout() to print numbers, but you don't have the double quotes
/*
int main()
{
	cout << 3+3 << "\n";
	// can also have more than one insertion operator
	cout << "Hello World" << "\n";
	cout << "Have a nice" << "day \n" ;
	//a way of inserting a new line is with endl
	cout << "This is a new line" << endl;
	cout << "I'm learning";
	return 0;
}
*/

//declaring variables

/*
int main()
{
	int num = 15;
	string name = "John";
	cout << num << "\n";
	cout << name << " is " << num << " years of age \n";

	// trying to calculate the area of a rectangle:

	int length = 6;
	int width = 4;
	int area = length*width;
	cout << "The area of the rectangle is " << area << endl;

	return 0;
}
*/

// trying to get user input

/*
int main() 
{
	// cin get's user input, using the extractor operator >>
	int x;
	cout << "Please type a number \n";
	cin >> x; // get user input and put it into x
	cout << "your number is " << x << endl;
}
*/

// can be lazy and use auto instead of explicitly declaring data types
// but you have to assign a value to the variable you can't just right
// auto x = 5;


int main()
{
	int studentID = 7;
	float score = 5.5;
	char grade = 'A';
	string greeting  = "Hello";
	cout << greeting << "\n";
	cout << studentID << " " << score << " " << grade << "\n";
}

