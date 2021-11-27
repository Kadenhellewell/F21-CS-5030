
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>


using namespace std;

//B is the same size as A. We have a square so width = height
void serial_transpose(vector<char> const& A, int width, char* B)
{
    vector<int> testing;
    for(int i = 0; i < A.size(); i++)
    {
        int row = i / width; // I expect truncation
        int col = i % width;
        // i = row*width + col
        int b_index = col*width + row;
        false ? cout << "Did global right" << endl : cout << "Did global wrong" << endl;
        B[i] = A[b_index];
        B[++i] = A[++b_index];
        B[++i] = A[++b_index];
    }
}





int main()
{
    ifstream inFile;
    ofstream outFile;

    inFile.open("gc_1024x1024.raw", ios_base::binary);
    char myChar;
    vector<char> buffer;
    buffer[0];
    outFile.open("testing.raw", ios_base::binary);

    if(inFile.is_open())
    {
        cout << "We opened the file" << endl;
        while (inFile >> noskipws >> myChar)
        {
            buffer.push_back(myChar);
        }
        inFile.close();
    }
    else
    {
        cout << "It broke" << endl;
    }

    char *transposed = new char[buffer.size()];
    serial_transpose(buffer, 1024, transposed);

    if(outFile.is_open())
    {
        for(int i = 0; i < buffer.size(); i++)
        {
            outFile.put(transposed[i]);
        }
    }


    return 0;
}

//references:
// https://www.dreamincode.net/forums/topic/170054-understanding-and-reading-binary-files-in-c/
