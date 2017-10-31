! module to store feature info

module feature_config

    !==============================!
    !* data structure definitions *!
    !==============================!

    !-------------------------------!
    !* types for distances, angles *!
    !-------------------------------!

    type,public :: feature_info_isotropic
        integer :: n                        ! number of neighbouring atoms
        real(8),allocatable :: dr(:)        ! distance to neighbour
        integer,allocatable :: idx(:)       ! index of neighbour
        real(8),allocatable :: drdri(:,:)   ! derivative of distance wrt atom
        real(8),allocatable :: z(:)         ! atomic number of neighbour
    end type feature_info_isotropic

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
        real(8) :: rcut                     ! interaction cut off 
        real(8) :: rs                       ! iso exp offset
        real(8) :: fs                       ! tapering smoothness
        real(8) :: eta                      ! iso
        real(8) :: xi                       ! ani
        real(8) :: lambda                   ! ani
        real(8) :: za                       ! central atomic number
        real(8) :: zb                       ! neighbour atomic number
    end type feature_

    type,public :: feature_info
        type(feature_),allocatable,public :: info(:)
        integer :: num_features
        logical :: pca
        real(8) :: pca_threshold
    end type feature_info

    !===================!
    !* initialisations *!
    !===================!

    !* feature parameters meta data    
    type(feature_info) :: feature_params

    !* atom distance-distance information
    type(feature_info_isotropic),allocatable :: feature_isotropic(:)

end module feature_config
