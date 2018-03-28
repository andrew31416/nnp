! module to store feature info

module feature_config

    !==============================!
    !* data structure definitions *!
    !==============================!

    !-------------------------------!
    !* types for distances, angles *!
    !-------------------------------!

    type,public :: feature_info_twobody
        integer :: n                        ! number of 2-body terms
        real(8),allocatable :: dr(:)        ! distance to neighbour
        integer,allocatable :: idx(:)       ! index of neighbour
        real(8),allocatable :: drdri(:,:)   ! derivative of distance wrt atom
        real(8),allocatable :: z(:)         ! atomic number of neighbour
        real(8) :: z_atom                   ! central atomic number
    end type feature_info_twobody

    type,public :: feature_info_threebody
        integer :: n                            ! number of 3-body terms
        real(8),allocatable :: cos_ang(:)       ! dtheta_{ijk}
        real(8),allocatable :: dr(:,:)          ! displacements between atoms
        real(8),allocatable :: z(:,:)           ! atomic number of neighbours
        integer,allocatable :: idx(:,:)         ! identifier of neighbours
        real(8) :: z_atom                       ! atomic number of central atom
        real(8),allocatable :: dcos_dr(:,:,:)   ! derivative of cos(angle) wrt atoms
        real(8),allocatable :: drdri(:,:,:)     ! derivative of atom-atom displacements
    end type feature_info_threebody

    !------------------------------!
    !* types for feature metadata *!
    !------------------------------!
    
    type,public :: feature_
        !=======================================================!
        ! ftype : 0 = atomic number                             !
        !         1 = Behler isotorpic                          !
        !         2 = Behler anisotropic                        !
        !         3 = normal isotorpic                          !
        !         4 = normal anisotropic                        !
        !=======================================================!
        integer :: ftype                    ! feature type
        real(8) :: rcut     = 0.0d0         ! interaction cut off 
        real(8) :: rs       = 0.0d0         ! iso exp offset
        real(8) :: fs       = 0.0d0         ! tapering smoothness
        real(8) :: eta      = 0.0d0         ! iso
        real(8) :: xi       = 0.0d0         ! ani
        real(8) :: lambda   = 0.0d0         ! ani
        real(8) :: sqrt_det = 0.0d0         ! precision matrix determinant
        real(8), allocatable :: prec(:,:)   ! normal precision
        real(8), allocatable :: mean(:)     ! normal mean
        real(8) :: za       = 0.0d0         ! central atomic number
        real(8) :: zb       = 0.0d0         ! neighbour atomic number
        real(8) :: scl_cnst = 1.0d0         ! scale all instances of this feature upon computation
        real(8) :: add_cnst = 0.0d0         ! additive constant for this feature upon computation
        real(8), allocatable :: devel(:)    ! for developing functions
    end type feature_

    type,public :: feature_info
        type(feature_),allocatable,public :: info(:)
        integer :: num_features = 0
        logical :: pca
        real(8) :: pca_threshold
    end type feature_info

    !===================!
    !* initialisations *!
    !===================!

    !* feature parameters meta data    
    type(feature_info),save :: feature_params

    !* two body information for a single structure used to generate features
    type(feature_info_twobody),allocatable :: feature_isotropic(:)

    !* three body information for a single structure used to generate features
    type(feature_info_threebody),allocatable :: feature_threebody_info(:)

    !* openMP pragma necessary for globally scoped variables
    !$omp threadprivate(feature_isotropic)
    !$omp threadprivate(feature_threebody_info)

    contains

        integer function featureID_StringToInt(feature_descriptor)
            !=======================================================!
            ! convert a string identifier for symmetry function to  !
            ! an integer identifier                                 !
            !=======================================================!

            implicit none

            character(len=*),intent(in) :: feature_descriptor
            
            integer :: idx

            idx = -1

            if (feature_descriptor.eq."atomic_number") then
                idx = 0
            else if (feature_descriptor.eq."acsf_behler-g1") then
                idx = 1
            else if (feature_descriptor.eq."acsf_behler-g2") then
                idx = 2
            else if (feature_descriptor.eq."acsf_behler-g4") then
                idx = 3
            else if (feature_descriptor.eq."acsf_behler-g5") then
                idx = 4
            else if (feature_descriptor.eq."acsf_normal-b2") then
                idx = 5
            else if (feature_descriptor.eq."acsf_normal-b3") then
                idx = 6
            else if (feature_descriptor.eq."devel_iso") then
                idx = 7
            else
                write(*,*) ""
                write(*,*) '*********************************************' 
                write(*,*) 'error raised in routine : feature_StringToInt'
                write(*,*) '*********************************************' 
                write(*,*) ""
                write(*,*) 'Error : ',"unrecognised feature type ",feature_descriptor
                write(*,*) ""
                call exit(0)
            end if
            featureID_StringToInt = idx
        end function featureID_StringToInt

        logical function feature_IsTwoBody(ftype)
            implicit none

            integer,intent(in) :: ftype
        
            logical :: tmp

            if ( (ftype.eq.1).or.(ftype.eq.2).or.(ftype.eq.5).or.(ftype.eq.7) ) then
                tmp = .true.
            else
                tmp = .false.
            end if
            feature_IsTwoBody = tmp
        end function
end module feature_config
