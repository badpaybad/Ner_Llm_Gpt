#include <iostream>
#include <memory>
#include <thread>
#include <fstream>
/*
cd ~

git clone https://github.com/nlohmann/json.git

cd json
mkdir build && cd build
cmake ..
sudo make install
cd ~
*/
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <stdio.h>
#include <tensorflow/c/c_api.h>

#include "libs/cloud/aws.h"
#include "libs/test.h"

int main()
{
    bool a = true;
    bool b(){
        return false;
    }
    while ()

        printf("Hello from TensorFlow C library version %s\n", TF_Version());
    
    return 0;
}
void testTryFinally()
{
    try
    {
        auto a = 0;
        auto b = 1;
        auto c = b / a;
        return;
    }
    catch (const std::exception &ex)
    {
        std::cout << "\nERRR\n";
        std::cout << ex.what();
    }
    catch (...)
    {
        std::cout << "\nwork like finally\n";
    }
}
void test2()
{
    // TestClass cls1("cls1");
    // cls1.funcPublic();

    // cloud::test();

    // try
    // {

    //     std::ifstream f("/work/llm/Ner_Llm_Gpt/cpplearning/test.json");
    //     json data = json::parse(f);

    //     std::cout << data << "\n";

    //     data["name"] = data.dump();

    //     std::cout << data.dump() << "\n";

    //     json xxx = {};

    //     xxx["du"] = data.dump();

    //     xxx["xxx"]["hello"] = "Hello";

    //     json tobj = {
    //         {"a", "abc"}};

    //     xxx["xxx"]["obj"] = tobj;

    //     std::cout
    //         << xxx.dump() << "\n";

    //     std::cout << "This is the line number " << __LINE__;
    //     std::cout << " of file " << __FILE__ << ".\n";
    //     std::cout << "Its compilation began " << __DATE__;
    //     std::cout << " at " << __TIME__ << ".\n";
    //     std::cout << "The compiler gives a __cplusplus value of " << __cplusplus;
    //     std::cout << "\n";
    // }
    // catch (...)
    // {
    // }
}
void test1()
{
    // std::cout << "Begin\n";

    // TestClass cls1("cls1");
    // cls1.funcPublic();

    // TestClass *cls2 = &cls1;
    // cls2->funcPublic();

    // cls2->Name = "cls2";

    // std::shared_ptr<TestClass> cls3 = std::make_shared<TestClass>(*cls2);
    // std::shared_ptr<TestClass> cls4 = std::make_shared<TestClass>(cls1);

    // //  cls4->Name = "cls4";
    // cls4->funcPublic();

    // cls3->Name = "cls3";
    // cls3->funcPublic();

    // cls1.funcPublic();
    // cls2->funcPublic();
    // testTryFinally();

    // std::cout << "End\n";
}