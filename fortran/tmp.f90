program main
    implicit none

    real(8),allocatable :: a(:,:)

    integer :: b

    allocate(a(5,4))
    write(*,*) shape(a)

    b = shape(a)(1)

    write(*,*) b
end program
