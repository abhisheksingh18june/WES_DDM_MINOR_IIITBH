from abc import ABC, abstractmethod

'''
The abstractmethod decorator is used to mark methods as abstract in an abstract base class (ABC).
It is used to specify that a method must be implemented in a subclass.
If it won't be implemented in a subclass, it will raise a NotImplementedError.
'''


class Animal(ABC):
    @abstractmethod
    def sound(self):
        """Abstract method for making a sound."""
        pass

    @abstractmethod
    def move(self):
        """Abstract method for movement."""
        pass


class Dog(Animal):
    def sound(self):
        return "Woof!"

    def move(self):
        return "Run"

class Bird(Animal):
    def sound(self):
        return "Chirp!"

    def move(self):
        return "Fly"


dog = Dog()
bird = Bird()

print(dog.sound())  
print(dog.move())   
print(bird.sound()) 
print(bird.move())  


