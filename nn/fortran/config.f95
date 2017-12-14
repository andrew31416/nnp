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
        real(8),dimension(:,:),allocatable :: hl1
        real(8),dimension(:,:),allocatable :: hl2
    end type unit_

    type,public :: units
        type(unit_),public :: a
        type(unit_),public :: a_deriv
        type(unit_),public :: z
        type(unit_),public :: delta
    end type units

    type,public :: feature_derivatives
        integer,allocatable :: idx(:)
        integer :: n
        real(8),allocatable :: vec(:,:)
    end type feature_derivatives

    !* atomic structure type
    type,public :: structure
        integer :: n                                            !* number of atoms
        real(8),dimension(:,:),allocatable :: r                 !* cartesian coordinates (3,n) / (A)
        real(8),dimension(:),allocatable :: z                   !* atomic number   
        real(8),dimension(1:3,1:3) :: cell                      !* cell vectors / (A) 
        real(8),dimension(:,:),allocatable :: x                 !* features (D+1,n)
        type(feature_derivatives),allocatable :: x_deriv(:,:)   !* feature derivatives (D,n)
        real(8),dimension(:),  allocatable :: current_ei        !* current per atom energies
        real(8),dimension(:,:),allocatable :: current_fi        !* current per atom forces
        real(8),dimension(:,:),allocatable :: ref_fi            !* ref forces
        real(8) :: ref_energy                                   !* ref total energy
    end type structure

    type,public :: structures
        type(structure),dimension(:),allocatable :: configs
        integer :: nconf = 0                            !* number of structures
    end type structures

    !----------------!
    ! initialisation !
    !----------------!

    !* net weights
    type(weights)  ,public,save :: net_weights
    type(weights)  ,public,save :: net_weights_nobiasT

    !* total number of net weights
    integer,public :: nwght

    !* dy/dw
    type(weights)  ,public,save :: dydw
    
    !* number of nodes per layer
    type(num_nodes),public,save :: net_dim
    
    !* type of nonlinear activation function
    integer,public :: nlf                               !* 1 = logistic, 2 = tanh 

    !* activation and backprop units
    type(units),public,save :: net_units

    !* dimension of features
    integer,public :: D

    !* data sets
    type(structures),public :: data_sets(1:2)

    !* derivative of net wrt features
    real(8),allocatable,public :: dydx(:,:)

    !* type of loss norm (l1 or l2)
    integer,public :: loss_norm_type = 1

    !* loss constants
    real(8),public :: loss_const_energy
    real(8),public :: loss_const_forces
    real(8),public :: loss_const_reglrn

    !* (forward and back)-prop behaviour
    logical,public :: calc_feature_derivatives = .true.
end module
