module config
    implicit none


    !* private scope by default
    private

    !* net weights
    type,public :: weights
        real(8),dimension(:,:),allocatable :: hl1       !* hidden layer 1
        real(8),dimension(:,:),allocatable :: hl2       !* hidden layer 2
        real(8),dimension(:),allocatable   :: hl3       !* hidden layer 3
    end type weights

    type,public :: num_nodes
        integer :: hl1
        integer :: hl2
    end type num_nodes

    type,public :: unit_
        real(8),dimension(:),allocatable :: hl1
        real(8),dimension(:),allocatable :: hl2
    end type unit_

    type,public :: units
        type(unit_),public :: a
        type(unit_),public :: delta
    end type units

    !* atomic structures
    type,public :: structure
        integer :: n                                    !* number of atoms
        real(8),dimension(:,:),allocatable :: x         !* features (D,n)
        real(8),dimension(:,:),allocatable :: forces    !* ref forces
        real(8) :: energy                               !* ref total energy
    end type structure

    !----------------!
    ! initialisation !
    !----------------!

    !* net weights
    type(weights)  ,public,save :: net_weights
    
    !* number of nodes per layer
    type(num_nodes),public,save :: net_dim
    
    !* type of nonlinear activation function
    integer,public :: nlf                               !* 1 = sigmoid, 2 = tanh 

    !* activation and backprop units
    type(units),public,save :: net_units

    !* dimension of features
    integer,public :: D

    !* train and test sets
    type(structure),public,dimension(:),allocatable :: train_set
    type(structure),public,dimension(:),allocatable :: test_set

end module
